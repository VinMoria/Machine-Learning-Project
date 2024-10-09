import pandas as pd
from finvizfinance.screener.overview import Overview


Sectors_list = ['Healthcare', 'Basic Materials', 'Financial', 'Consumer Defensive','Industrials', 'Technology', 'Consumer Cyclical', 'Real Estate','Communication Services', 'Energy', 'Utilities']

industry_ticker_dict={}
for sector in Sectors_list:
    print(f"\n\n\n<<<<<<<<<< Sector: {sector} started >>>>>>>>>>\n\n\n")
    stock_list = Overview()
    stock_list.set_filter(filters_dict={"Sector": sector})
    stock_data = stock_list.screener_view()
    tickers_list = list(stock_data.Ticker)
    industry_ticker_dict[sector] = tickers_list

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in industry_ticker_dict.items()]))
df.to_csv('industry_stocks.csv', index=False)