from ChatAnalysis import ChatAnalysis
import os


if __name__ == '__main__':
    output_dir = 'charts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    chat_analysis = ChatAnalysis('demo_utf8.csv', output_dir, sender_name='帅先生', receiver_name='美小姐')
    chat_analysis.preprocess_data()  # 数据预处理
    chat_analysis.plot_chat_frequency_by_day()  # 绘制每天聊天频率的柱状图并保存
    chat_analysis.plot_word_frequency(top_n=20)  # 绘制词频分析图并保存
    chat_analysis.plot_chat_frequency_by_hour()
    chat_analysis.plot_chat_comparison()
    chat_analysis.plot_monthly_chat_frequency_comparison()
    chat_analysis.plot_weekly_chat_heatmap()
    chat_analysis.sentiment_analysis()
    chat_analysis.analyze_interaction_patterns()
    chat_analysis.generate_html_report(output_dir, "chat_analysis_template.html")
