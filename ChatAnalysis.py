import os
from string import Template

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from snownlp import SnowNLP

class ChatAnalysis:
    def __init__(self, csv_file, output_dir='charts', sender_name='Sender 1', receiver_name='Sender 0'):
        self.df = pd.read_csv(csv_file)
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        self.output_dir = output_dir
        self.sender_name = sender_name
        self.receiver_name = receiver_name

    def preprocess_data(self, chat_type=1, columns=['IsSender', 'StrContent', 'StrTime']):
        self.df = self.df[self.df['Type'] == chat_type]
        self.df = self.df[columns]
        self.df['StrTime'] = pd.to_datetime(self.df['StrTime'])
        self.df['Date'] = self.df['StrTime'].dt.date
        self.df['Month'] = self.df['StrTime'].dt.strftime('%Y-%m')
        self.df['TimeDiff'] = self.df.groupby('Date')['StrTime'].diff().dt.total_seconds().div(60)  # Minutes

        # 新增计算是否为对话发起和计算对话块ID的逻辑
        self.df['PrevSender'] = self.df['IsSender'].shift()
        self.df['BlockStart'] = (self.df['IsSender'] != self.df['PrevSender']).astype(int)
        self.df['BlockId'] = self.df['BlockStart'].cumsum()
        # 为回复时间分析添加所需的列
        self.df['NextStrTime'] = self.df['StrTime'].shift(-1)
        self.df['ReplyTime'] = np.where(self.df['IsSender'] != self.df['PrevSender'], (self.df['NextStrTime'] - self.df['StrTime']).dt.total_seconds().div(60), np.nan)
        self.df['IsReplyToOther'] = self.df['IsSender'] != self.df['PrevSender']

    def plot_chat_frequency_by_day(self):
        plt.figure()  # 创建新画布
        chat_frequency = self.df['Date'].value_counts().sort_index()
        chat_frequency.plot(kind='bar', color='#DF9F9B')
        plt.xlabel('Date')
        plt.ylabel('Frequency')
        plt.title('Daily Chat Frequency')
        date_labels = [date.strftime('%m-%d') for date in chat_frequency.index]
        plt.xticks(range(len(date_labels)), date_labels, fontsize=5)
        plt.savefig(f'{self.output_dir}/chat_frequency_by_day.png')

    def plot_chat_frequency_by_hour(self):
        plt.figure()  # 创建新画布
        self.df['Hour'] = self.df['StrTime'].dt.hour
        hourly_counts = self.df['Hour'].value_counts().sort_index().reset_index()
        hourly_counts.columns = ['Hour', 'Frequency']
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Hour', y='Frequency', data=hourly_counts, color="#E6AAAA")
        sns.kdeplot(self.df['Hour'], color='#C64F4F', linewidth=1, ax=ax.twinx())
        plt.title('Hourly Chat Frequency')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Frequency')
        plt.savefig(f'{self.output_dir}/chat_frequency_by_hour.png')

    def plot_word_frequency(self, top_n=20):
        plt.figure(figsize=(10, 6))  # 创建新画布

        # 过滤掉包含'['的聊天记录
        sent_by_me = self.df[self.df['IsSender'] == 1]['StrContent'].apply(lambda x: x if '[' not in x else '')
        sent_by_others = self.df[self.df['IsSender'] == 0]['StrContent'].apply(lambda x: x if '[' not in x else '')

        # 将过滤后的聊天记录转换为小写并分割为单词
        words_sent_by_me = ' '.join(sent_by_me).lower().split()
        words_sent_by_others = ' '.join(sent_by_others).lower().split()

        # 统计词频
        word_freq_sent_by_me = pd.Series(words_sent_by_me).value_counts()
        word_freq_sent_by_others = pd.Series(words_sent_by_others).value_counts()

        # 绘制我发送的聊天信息词频图
        plt.subplot(121)
        word_freq_sent_by_me[:top_n].plot(kind='barh', color='#009688')
        plt.title(f'Word Frequency (Sent by {self.sender_name})')
        plt.xlabel('Frequency')

        # 绘制对方发送的聊天信息词频图
        plt.subplot(122)
        word_freq_sent_by_others[:top_n].plot(kind='barh', color='#6A0404')
        plt.title(f'Word Frequency (Sent by {self.receiver_name})')
        plt.xlabel('Frequency')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/word_frequency.png')

    def plot_chat_comparison(self):
        plt.figure()  # 创建新画布
        labels = [self.sender_name, self.receiver_name]
        sizes = [len(self.df[self.df['IsSender'] == 1]), len(self.df[self.df['IsSender'] == 0])]
        colors = ['#264F61', '#9523AA']
        explode = (0, 0.05)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')
        plt.title('Comparison of the Number of Chats')
        plt.legend()
        plt.savefig(f'{self.output_dir}/chat_comparison.png')

    def plot_monthly_chat_frequency_comparison(self):
        plt.figure()  # 创建新画布
        monthly_chat = self.df.groupby(['Month', 'IsSender']).size().unstack().fillna(0)
        monthly_chat.columns = [self.receiver_name, self.sender_name]  # 更新列名为参与者代称
        monthly_chat.plot(kind='bar', figsize=(10, 6))
        plt.xlabel('Month')
        plt.ylabel('Number of Messages')
        plt.title('Monthly Chat Frequency Comparison')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/monthly_chat_frequency_comparison.png')


    def plot_weekly_chat_heatmap(self):
        # 确保'StrTime'列为datetime类型
        self.df['StrTime'] = pd.to_datetime(self.df['StrTime'])
        self.df['Date'] = self.df['StrTime'].dt.date

        # 新增'WeekDay'和'WeekNumber'列
        self.df['WeekDay'] = self.df['StrTime'].dt.dayofweek + 1  # 周一为1，周日为7
        self.df['WeekNumber'] = self.df['StrTime'].dt.isocalendar().week

        # 对每周的每天进行聊天记录数量统计
        result_total_day = self.df.groupby(['WeekNumber', 'WeekDay'])['StrContent'].count().reset_index()

        # 对result_total_day数据进行处理
        msg_dict = dict()
        for week, week_df in result_total_day.groupby('WeekNumber'):
            week_data = week_df.set_index('WeekDay')['StrContent'].reindex(range(1, 8), fill_value=0).values
            msg_dict[week] = week_data

        # 准备热力图数据
        y_labels = list(msg_dict.keys())
        values = np.array(list(msg_dict.values()))
        x_labels = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

        # 绘图
        fig, axe = plt.subplots(figsize=(15, 15))  # size可以调整
        axe.set_xticks(np.arange(len(x_labels)))
        axe.set_yticks(np.arange(len(y_labels)))
        axe.set_xticklabels(x_labels)
        axe.set_yticklabels(y_labels)
        im = axe.imshow(values, cmap="YlGnBu")  # 颜色更改为Yellow-Green-Blue渐变
        # 添加色标以提供参考
        plt.colorbar(im)
        plt.savefig(f'{self.output_dir}/weekly_chat_heatmap.png')

    def sentiment_analysis(self):
        # 确保'StrTime'列为datetime类型，'StrContent'为聊天内容
        self.df['StrTime'] = pd.to_datetime(self.df['StrTime'])
        self.df['Month'] = self.df['StrTime'].dt.strftime('%Y-%m')

        # 对每条消息进行情感分析
        self.df['Sentiment'] = self.df['StrContent'].apply(lambda x: SnowNLP(x).sentiments)

        # 根据情感得分将情绪分为积极、消极、中性三类
        self.df['Emotion'] = pd.cut(self.df['Sentiment'], bins=[0, 0.4, 0.6, 1],
                                    labels=['Negative', 'Neutral', 'Positive'])

        # 按月份、发送者、情绪类别统计
        self.df['SenderType'] = self.df['IsSender'].apply(lambda x: 'Sender' if x == 1 else 'Receiver')
        emotion_counts = self.df.pivot_table(index=['Month', 'SenderType'], columns='Emotion', aggfunc='size',
                                             fill_value=0)

        # 重塑DataFrame以便作图
        emotion_counts = emotion_counts.reset_index()

        # 绘制累计柱状图
        plt.figure(figsize=(10, 6))
        for sender_type, group in emotion_counts.groupby('SenderType'):
            index = range(len(group['Month'])) if sender_type == 'Sender' else [x + 0.4 for x in
                                                                                range(len(group['Month']))]
            plt.bar(index, group['Positive'], width=0.4, label=f'Positive ({sender_type})',
                    bottom=group['Neutral'] + group['Negative'])
            plt.bar(index, group['Neutral'], width=0.4, label=f'Neutral ({sender_type})', bottom=group['Negative'],
                    color='gray')
            plt.bar(index, group['Negative'], width=0.4, label=f'Negative ({sender_type})')

        plt.xticks([x + 0.2 for x in range(len(emotion_counts['Month'].unique()))], emotion_counts['Month'].unique(),
                   rotation=45)
        plt.xlabel('Month')
        plt.ylabel('Number of Messages')
        plt.title('Monthly Emotional Distribution by Sender Type')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/monthly_emotional_distribution_combined.png')

    def analyze_interaction_patterns(self):
        # 替换IsSender列中的0和1为接收者和发送者的名字
        sender_mapping = {0: self.receiver_name, 1: self.sender_name}
        self.df['IsSender'] = self.df['IsSender'].map(sender_mapping)
        # 对话发起者
        self.df['Initiates'] = (self.df['TimeDiff'] > 60) | (self.df['TimeDiff'].isna())
        initiator_daily = self.df.groupby(['Date', 'IsSender', 'Initiates', 'Month']).size().reset_index(name='Counts')
        initiator_monthly = initiator_daily[initiator_daily['Initiates']].groupby(['Month', 'IsSender'])[
            'Counts'].count().unstack(fill_value=0)

        # 连续聊天的话语数量
        self.df['Block'] = (self.df['IsSender'] != self.df['IsSender'].shift()).cumsum()
        messages_per_block_daily = self.df.groupby(['Date', 'Block', 'IsSender', 'Month']).size().reset_index(
            name='MessageCount')
        avg_messages_per_block_monthly = messages_per_block_daily.groupby(['Month', 'IsSender'])[
            'MessageCount'].mean().unstack(fill_value=0)

        # 回复等待时间
        self.df['ReplyTime'] = np.where(self.df['IsSender'] != self.df['IsSender'].shift(), self.df['TimeDiff'], np.nan)
        avg_reply_time_daily = self.df.groupby(['Date', 'IsSender', 'Month'])['ReplyTime'].mean().reset_index()
        avg_reply_time_monthly = avg_reply_time_daily.groupby(['Month', 'IsSender'])['ReplyTime'].mean().unstack(
            fill_value=0)

        # 绘图
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

        # 对话发起者图
        initiator_monthly.plot(kind='bar', ax=axes[0], title='Monthly Conversation Initiations')
        # 设置图例
        axes[0].legend([self.sender_name, self.receiver_name], title='发起会话者')

        # 连续聊天的话语数量图
        avg_messages_per_block_monthly.plot(kind='bar', ax=axes[1],
                                            title='Average Messages Per Conversation Block Monthly')
        # 设置图例
        axes[1].legend([self.sender_name, self.receiver_name], title='说话人')

        # 回复等待时间图
        avg_reply_time_monthly.plot(kind='bar', ax=axes[2], title='Average Reply Time Monthly')
        # 设置图例
        axes[2].legend([self.sender_name, self.receiver_name], title='等待其回复的时间')

        axes[0].set_ylabel('Initiation Count')
        axes[1].set_ylabel('Average Message Count')
        axes[2].set_ylabel('Average Reply Time (min)')
        axes[2].set_xlabel('Month')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/interaction_patterns.png')

    def generate_html_report(self, output_dir, template_file):
        # 图表信息：标题、文件名和描述
        chart_info = [
            {
                "title": "Chat Comparison",
                "file": "chat_comparison.png",
                "description": "Comparison of the number of messages sent by each participant."
            },
            {
                "title": "Daily Chat Frequency",
                "file": "chat_frequency_by_day.png",
                "description": "Frequency of chats for each day."
            },
            {
                "title": "Hourly Chat Frequency",
                "file": "chat_frequency_by_hour.png",
                "description": "Distribution of chats throughout the day."
            },
            {
                "title": "Interaction Patterns",
                "file": "interaction_patterns.png",
                "description": "Patterns of how participants interact in the conversation."
            },
            {
                "title": "Monthly Chat Frequency Comparison",
                "file": "monthly_chat_frequency_comparison.png",
                "description": "Monthly comparison of the number of messages sent by each participant."
            },
            {
                "title": "Monthly Emotional Distribution",
                "file": "monthly_emotional_distribution_combined.png",
                "description": "Emotional analysis of messages on a monthly basis."
            },
            {
                "title": "Weekly Chat Heatmap",
                "file": "weekly_chat_heatmap.png",
                "description": "Heatmap showing the distribution of chats throughout the week."
            },
            {
                "title": "Word Frequency",
                "file": "word_frequency.png",
                "description": "Most common words used in the conversation."
            },
        ]

        charts_html = ""
        for chart in chart_info:
            charts_html += f"""
            <div class="chart">
                <h2 class="chart-title">{chart['title']}</h2>
                <p class="chart-description">{chart['description']}</p>
                <img src="{output_dir}/{chart['file']}" class="img-fluid">
            </div>
            """

        # 从模板文件读取 HTML 模板
        with open(template_file, 'r', encoding='utf-8') as file:
            src = Template(file.read())

        # 替换模板中的占位符
        html_content = src.substitute(charts=charts_html)

        # 保存 HTML 报告
        report_path = "chat_analysis_report.html"
        with open(report_path, 'w', encoding='utf-8') as file:
            file.write(html_content)

        print(f"Report generated: {report_path}")
