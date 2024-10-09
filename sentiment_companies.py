import pandas as pd 
import sentiment

def read_companies(sector):
    sector_csv = pd.read_csv(f"train_set/{sector}.csv")
    quarter_company = sector_csv[['Quarter', 'Ticker']]
    y = quarter_company.head(5)

    results_list = []

    for index, row in y.iterrows():
        

        row.Quarter = pd.to_datetime(row.Quarter)
        start_date = row.Quarter - pd.DateOffset(months=1)
        end_date = row.Quarter + pd.DateOffset(months=1)

        # convert date into string
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        scores = sentiment.get_stock_sentiment(row.Ticker, start_date, end_date)
        
        results_list.append({'Quarter': row.Quarter, 'Ticker': row.Ticker, 'Sentiment_Score': scores})

    results_df = pd.DataFrame(results_list)

    return results_df

print(read_companies('Basic Materials'))