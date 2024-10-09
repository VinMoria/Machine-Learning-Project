import pandas as pd 
import sentiment
import os

def read_companies(sector):
    sector_csv = pd.read_csv(f"train_set/{sector}.csv")
    sector_csv = sector_csv.head(5)
    quarter_company = sector_csv[['Quarter', 'Ticker']]
    

    results_list = []

    for index, row in quarter_company.iterrows():
        print(f"{index}/{quarter_company.shape[0]}, Quarter:{row.Quarter}, Ticker:{row.Ticker}")

        row.Quarter = pd.to_datetime(row.Quarter)
        start_date = row.Quarter - pd.DateOffset(months=1)
        end_date = row.Quarter + pd.DateOffset(months=1)

        # convert date into string
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        scores = sentiment.get_stock_sentiment(row.Ticker, start_date_str, end_date_str)
        
        # results_list.append({'Quarter': row.Quarter, 'Ticker': row.Ticker, 'Sentiment_Score': scores})
        print(scores)
        results_list.append(scores)

    # results_df = pd.DataFrame(results_list)
    print(results_list)

    sector_csv.insert(sector_csv.shape[1], 'Sentiment_Score', results_list)

    output_file = f"add_sentiment_train_set/{sector}_sentiment.csv"

    sector_csv.to_csv(output_file, index=False)
    
    return sector_csv

print(read_companies('Basic Materials'))