import pandas as pd
import yfinance as yf
from finvizfinance.screener.overview import Overview
from datetime import datetime, timedelta
import numpy as np
import time
import random

N_TICKER_IN_A_SECTOR = 200
SLEEP_TIME = 0.5

# 获取市值，由get_finace_df调用
def get_ticker_market_cap(ticker, date):
    try:
        ticker_obj = yf.Ticker(ticker)
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        new_date_obj = date_obj + timedelta(days=10)
        end_date = new_date_obj.strftime('%Y-%m-%d')
        historical_data = ticker_obj.history(start=date, end=end_date)
        shares_outstanding = ticker_obj.info['sharesOutstanding']
        return float(historical_data['Close'].iloc[0] * shares_outstanding / 1000000)
    except:
        return -1

# 获取每个ticker的df
def get_finace_df(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 获取季度财务报表
        balance_sheet_quarterly = stock.quarterly_balance_sheet  # 季度资产负债表
        income_statement_quarterly = stock.quarterly_financials  # 季度利润表
        cashflow_quarterly = stock.quarterly_cashflow            # 季度现金流量表

        def handle(df): # 转置，删除最后一行
            df = df.transpose()
            df = df.iloc[:min(5,df.shape[0])]
            return df
        balance_sheet_quarterly = handle(balance_sheet_quarterly)
        income_statement_quarterly = handle(income_statement_quarterly)
        cashflow_quarterly = handle(cashflow_quarterly)

        # 将3个df列合并
        df_merged = pd.merge(balance_sheet_quarterly, income_statement_quarterly, left_index=True, right_index=True)
        df_merged = pd.merge(df_merged, cashflow_quarterly, left_index=True, right_index=True)
        # 插入Ticker和Sector
        df_merged.insert(0, 'Ticker', [ticker]*df_merged.shape[0])
        df_merged = df_merged.reset_index()
        df_merged.rename(columns={'index': 'Quarter'}, inplace=True)
        # 插入市值（Y）
        quarter_list = list(df_merged.Quarter)
        marker_cap_list = []
        for quarter in quarter_list:
            quarter = quarter.strftime('%Y-%m-%d')
            marker_cap = get_ticker_market_cap(ticker, quarter)
            if marker_cap == -1:
                return -1
            else:
                marker_cap_list.append(marker_cap)
        df_merged.insert(df_merged.shape[1], 'Market Cap(M)', marker_cap_list)
        if df_merged.shape[0] == 0:
            return -1
        print(f"[ {ticker} ]: data shape - {df_merged.shape}")
        return df_merged
    except:
        return -1

df = pd.read_csv('industry_stocks.csv')
sector_dict = df.to_dict(orient='list')
for sector in sector_dict:
    sector_dict[sector] = [value for value in sector_dict[sector] if isinstance(value, str)]

for sector in sector_dict:
    print(f"\n<============== Sector: <{sector}> started ==============>")
    ticker_list = sector_dict[sector]
    df_sector_one_file_list = []
    ticker_count = 0
    while (ticker_count < N_TICKER_IN_A_SECTOR) and (len(ticker_list)>0):
        time.sleep(SLEEP_TIME)
        random_index = random.randint(0, len(ticker_list) - 1)
        ticker = ticker_list.pop(random_index)
        df_ticker = get_finace_df(ticker)
        if not isinstance(df_ticker, int):
            ticker_count += 1
            df_sector_one_file_list.append(df_ticker)
            print(f"{ticker_count}/{N_TICKER_IN_A_SECTOR}")
    df_sector_one_file = pd.concat(df_sector_one_file_list, ignore_index=True)
    filename = "./train_set/"+sector+".csv"
    df_sector_one_file.to_csv(filename, index=False)
    print(f"{sector}.csv saved")
