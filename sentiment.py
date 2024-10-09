import datetime
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import pandas as pd
import yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

def get_stock_sentiment(stock_code, start_date, end_date, url_sentiment="https://www.ft.com/search?expandRefinements=true&q={}&dateFrom={}&dateTo={}"):
    stock_info = yf.Ticker(stock_code)
    company_name = stock_info.info['longName']
    print(f"stock code: {stock_code} , company name：{company_name}")
    symbs = company_name
    #TODO date转为参数
    current_date = datetime.now().date()
    url_sentiments = [url_sentiment.format(symbs, start_date, end_date)]

    r = requests.get(url_sentiments[0])
    soup = BeautifulSoup(r.content, 'html.parser')
    data = soup.find_all("a", class_="js-teaser-heading-link")
    titles = []
    for index, each in enumerate(data): 
        each_title =each.text.strip() 
        titles.append(each_title)
        full_text = ' '.join(titles)

    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('vader_lexicon')

    analyzer = SentimentIntensityAnalyzer()
    # scores = analyzer.polarity_scores(full_text)
    # labels = list(scores.keys())
    # values = list(scores.values())
    # plt.figure(figsize=(8, 5))
    # colors = ['#A6B1B5', '#B7D9B5', '#D0C2D5', '#E7D9B2']
    # bars = plt.bar(labels, values, color=colors)

    # # 在每个柱子上显示数据值
    # for bar in bars:
    #     yval = bar.get_height()  # 获取柱子的高度
    #     plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', 
    #             ha='center', va='bottom')  # 在柱子上方显示数值

    # plt.title(symbs)
    # plt.xlabel("sentiment")
    # plt.ylabel("score")
    # plt.show()

    compound_score = analyzer.polarity_scores(full_text)['compound']
    # print(f"compound_score = {compound_score}")
    return compound_score
    
print(get_stock_sentiment("AAPL", "2024-05-01", "2024-06-01"))
