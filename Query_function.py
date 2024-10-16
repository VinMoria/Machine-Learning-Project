import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import os
import subprocess
import platform
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from wordcloud import WordCloud
import sentiment
MODEL_PATH = os.path.dirname(os.path.abspath(__file__))

def fetch_and_analyze_company(stock_code):
    stock = yf.Ticker(stock_code)
    df = stock.history(period="1y") # obtain historical data for the past 1 year
    if df.empty: # check whether df is empty
        messagebox.showerror("Error", "No data found for the specified stock code.")
        return None, None
    df.index = df.index.tz_localize(None)
    # Converts the timestamp index (date) in df from a format with time zone information to a format without time zones.
    # This step avoids potential problems due to time zones when processing data.
    return df, stock.info


def create_excel_with_analysis(stock_code, df, info): # "info" contains detailed data related to the stock.
    current_dir = os.getcwd()  # use os.getcwd() to get the current working directory and save the file
    filename = os.path.join(current_dir, f"{MODEL_PATH}/financial_report/{stock_code}_financial_report.xlsx") # generate the full path and filename of the Excel file
  
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name="Historical Data") # Write the Historical Data df of the stock to the "Historical Data" worksheet in the Excel file.
        plot_data(df, writer, "Historical Data", stock_code)  # Plot the stock price trend and put it into the "Historical Data" worksheet in the Excel file.
        insert_integrated_data(writer, info, "Integrated Data") # Write the financial data of the stock (such as PE ratio, profit margin, etc.) into the Integrated Data worksheet.
        insert_additional_data(writer, info, "Financial Report") # Write the company's Financial reporting information (such as dividend yield, earnings per share growth, etc.) into the Financial Report worksheet.
        insert_earnings_calendar(writer, stock_code, "Earnings Calendar") # Write the company's next Earnings date to the Earnings Calendar worksheet.
        insert_executive_team(writer, info, "Executive Team") # Write information about the company's Executive Team(name and title), into the Executive Team worksheet.
        insert_related_news(writer, stock_code, "Related News") # Fill in the Executive team worksheet with information about the company's executive team, such as name and title.
        insert_sentiment_analysis(writer, stock_code, "Sentiment")

    open_excel_file(filename)
    return filename


def get_financial_data(stock_code):
    stock = yf.Ticker(stock_code)
    earnings_date = stock.earnings_dates # Extract the stock's history and future earnings release dates.
    current_time = pd.Timestamp.now(tz='America/New_York') # Unified to New York time
    future_earnings = earnings_date[earnings_date.index > current_time] 
    # Filter future earnings dates from earnings_date, that is, dates greater than the current time.

    if not future_earnings.empty:
        next_earnings_date = future_earnings.index.min()
    else:
        next_earnings_date = None
    # If future_earnings is not empty, take the earliest date as the next earnings date.
    # If there is no future earnings date, set next_earnings_date to None.

    return next_earnings_date


def insert_integrated_data(writer, info, sheet_name):
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    # Define data segment
    sections = [
        ("Valuation Analysis", {
            "PE Ratio": (info.get("trailingPE", "N/A"), "Indicates the ratio of a company's current stock price to its earnings per share (EPS). A high PE Ratio suggests that investors have high expectations for the company's future growth, but it may also indicate overvaluation."),
            "PB Ratio": (info.get("priceToBook", "N/A"), "Measures the ratio of a company’s stock price to its book value per share. A higher PB Ratio may imply that the stock is overvalued relative to its net assets."),
            "PEG Ratio": (info.get("pegRatio", "N/A"), "Combines the PE ratio with the company’s future earnings growth rate. A PEG Ratio higher than 1 suggests that the company’s valuation may be overestimated.")
        }),
        ("Profitability", {
            "Profit Margins": (info.get("profitMargins", "N/A"), "Measures the percentage of net income that constitutes total revenue. Higher profit margins indicate that the company is performing well in controlling costs and generating profits."),
            "Earnings Per Share": (info.get("trailingEps", "N/A"), "Represents the company’s net income divided by the total number of outstanding shares. A higher EPS means that the company is generating more profit per share for its shareholders."),
            "EBITDA Margins": (info.get("ebitdaMargins", "N/A"), "Measures the profitability of a company’s core operations without considering interest, taxes, depreciation, and amortization. A higher EBITDA margin suggests strong profitability in the company’s core business.")
        }),
        ("Financial Situation", {
            "Debt to Equity Ratio": (info.get("debtToEquity", "N/A"), "Reflects the degree to which a company is financing its operations through debt. A higher ratio indicates reliance on debt financing, which may increase financial risk."),
            "Free Cashflow": (info.get("freeCashflow", "N/A"), "Represents the cash flow remaining after the company has accounted for capital expenditures. Positive free cash flow means the company has sufficient cash to pay dividends, repay debt, or invest in other projects."),
            "Operating Cashflow": (info.get("operatingCashflow", "N/A"), "Reflects the cash generated from the company’s core business operations. Positive operating cash flow indicates that the company is performing well and generating sufficient cash from its business activities.")
        })
    ]
    row = 1
    for title, metrics in sections: # Writes data to the worksheet
        worksheet.write(f'A{row}', title)
        worksheet.write(f'B{row}', "Data")
        worksheet.write(f'C{row}', "Explanation")
        row += 1
        for metric, (value, judgement) in metrics.items():
            worksheet.write(f'A{row}', metric)
            worksheet.write(f'B{row}', value)
            worksheet.write(f'C{row}', judgement)
            row += 1
        row += 1  # This line adds the empty row


def plot_data(df, writer, sheet_name, stock_code): # Plot the closing price trend of a stock
    fig, ax = plt.subplots(figsize=(10, 5))
    df['Close'].plot(ax=ax)
    ax.set_title("Stock Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    chart_path = f"{MODEL_PATH}/financial_report/{stock_code}_stock_plot.png"
    fig.savefig(chart_path)
    plt.close(fig)
    writer.sheets[sheet_name].insert_image('G1', chart_path)
    # Inserts the saved image into cell G1 in the specified sheet (sheet_name).


def open_excel_file(filename): # Open the finished excel file
    if platform.system() == "Windows":
        os.startfile(filename)
    elif platform.system() == "Darwin":
        subprocess.call(["open", filename])
    else:
        subprocess.call(["xdg-open", filename])


def user_interaction():
    root = tk.Tk()
    root.title("valuAItion Prototype / Invest Version")
    root.geometry("400x300")

    tk.Label(root, text="Enter Stock Code:").pack(pady=10)
    stock_code_var = tk.StringVar() # to store the stock symbol entered by the user
    stock_code_entry = ttk.Entry(root, textvariable=stock_code_var)
    # Create an input box for the user to enter the stock code and bind it to stock_code_var 
    # so that the values entered in the input box are automatically updated to stock_code_var
    stock_code_entry.pack(pady=10)

    ttk.Button(root, text="Fetch Data", command=lambda: execute_fetch(stock_code_var.get().strip())).pack(pady=20)
    # Create a button that displays the text "Fetch Data". When the button is clicked, the execute_fetch function is called, 
    # passing in the stock symbol that the user entered in the input box (obtained through stock_code_var.get()). 
    # Use the pack() method to add the button to the window and set the upper and lower spacing to 20 pixels.
    
    info_text = "Examples: \n" \
                "New York Stock Exchange: AAPL\n" \
                "Nasdaq: MSFT\n" \
                "Shanghai Stock Exchange: 600028.SS\n" \
                "Hong Kong Stock Exchange: 1810.HK\n" \
                "Tokyo Exchange: 7203.T"
    tk.Label(root, text=info_text, justify=tk.LEFT).pack(pady=5)

    root.mainloop()


def execute_fetch(stock_code): # Process stock symbols entered by users and perform data acquisition and analysis operations
    df, info = fetch_and_analyze_company(stock_code)
    if df is not None:
        filename = create_excel_with_analysis(stock_code, df, info)
        messagebox.showinfo("Success", f"Data has been successfully exported to Excel at {filename}.")


def insert_additional_data(writer, info, sheet_name): # Insert additional data related to the stock in the Excel file
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    sections = [
        ("Dividend Information", {
            "Dividend Yield": (info.get("dividendYield", "N/A"), "Dividend Yield indicates the ratio of dividends distributed by a company to its shareholders relative to its stock price. A higher dividend yield suggests that the company provides a relatively high return to shareholders, making it suitable for investors seeking stable income."),
            "Payout Ratio": (info.get("payoutRatio", "N/A"), "Payout Ratio measures the portion of net income that a company allocates to paying dividends to shareholders. A higher payout ratio indicates that the company is distributing more profit to shareholders rather than reinvesting it.")
        }),
        ("Earnings Growth", {
            "EPS Growth": (info.get("earningsGrowth", "N/A"), "Earnings Per Share (EPS) Growth Rate reflects the growth rate of a company's EPS compared to previous periods, indicating changes in the company's profitability. A higher EPS growth rate is generally seen as a sign of good financial health and enhanced profitability for the company."),
            "Target Mean Price": (info.get("targetMeanPrice", "N/A"), "Target Mean Price is the average predicted future stock price of a company as estimated by analysts.")
        }),
        ("Market Influence", {
            "Beta": (info.get("beta", "N/A"), "Beta is a measure of a stock's volatility relative to the overall market. A higher beta indicates that the stock has higher risk, but it may also offer greater potential returns."),
            "Held by Institutions": (info.get("heldPercentInstitutions", "N/A"), "Held by Institutions represents the proportion of a company's stock that is held by institutional investors (such as funds, pension plans, etc.)."),
            "Short Percentage of Float": (info.get("shortPercentOfFloat", "N/A"), "Short Percentage of Float denotes the percentage of shares that have been sold short relative to the total float.")
        })
    ]
    row = 1
    for title, metrics in sections:
        worksheet.write(f'A{row}', title)
        worksheet.write(f'B{row}', "Data")
        worksheet.write(f'C{row}', "Explanation")
        row += 1
        for metric, (value, judgement) in metrics.items():
            worksheet.write(f'A{row}', metric)
            worksheet.write(f'B{row}', value)
            worksheet.write(f'C{row}', judgement)
            row += 1
        row += 1  # This line adds the empty row

        
def insert_earnings_calendar(writer, stock_code, sheet_name):
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    worksheet.write('A1', 'Next Earnings Date(New York time)')

    earnings_date = get_financial_data(stock_code)

    if earnings_date is not None:
        # Convert earnings_date to a string formatted as 'YYYY-MM-DD_HHMMSS'
        earnings_date_str = time.strftime('%Y-%m-%d %H:%M:%S', earnings_date.timetuple())
        worksheet.write('A2', earnings_date_str)
    else:
        worksheet.write('A2', 'No upcoming earnings date found.')


def insert_executive_team(writer, info, sheet_name):
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    worksheet.write('A1', 'Name')
    worksheet.write('B1', 'Position')

    executives = info.get("companyOfficers", [])
    row = 2
    for executive in executives:
        worksheet.write(row, 0, executive.get('name', 'N/A'))
        worksheet.write(row, 1, executive.get('title', 'N/A'))
        row += 1

        
def insert_related_news(writer, stock_code, sheet_name):

    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    worksheet.write('A1', 'Title')
    worksheet.write('B1', 'Link')
    worksheet.write('C1', 'Published Time(New York time)')

    stock = yf.Ticker(stock_code)
    news = stock.news

    if news:
        # If there is a news item, the code will start writing the data for each news item, starting with the second line
        row = 2
        for item in news:
            title = item.get('title', 'N/A')
            link = item.get('link', 'N/A')
            publish_time = pd.to_datetime(item.get('providerPublishTime', 0), unit='s', utc=True)  # Convert the release time to UTC format
            
            # Converts UTC time to New York time
            publish_time_ny = publish_time.tz_convert('America/New_York')

            worksheet.write(row, 0, title)
            worksheet.write(row, 1, link)
            worksheet.write(row, 2, str(publish_time_ny))
            row += 1
    else:
        worksheet.write('A2', 'No news found.')


def insert_sentiment_analysis(writer, stock_code, sheet_name):
    workbook = writer.book
    worksheet = workbook.add_worksheet(sheet_name)
    writer.sheets[sheet_name] = worksheet

    # get the current date and the date one month ago
    current_date = datetime.now()
    current_year_month_day = current_date.strftime('%Y-%m-%d')
    one_month_ago = current_date - relativedelta(months=1)
    one_month_ago_year_month_day = one_month_ago.strftime('%Y-%m-%d')

    # call the function in sentiment.py
    sentiment_score, long_text = sentiment.get_stock_sentiment(stock_code, one_month_ago_year_month_day, current_year_month_day)

    worksheet.write('A1', 'Sentiment in Last 1 Month')
    worksheet.write('B1', str(sentiment_score))
    worksheet.write('A2', 'From')
    worksheet.write('B2', one_month_ago_year_month_day)
    worksheet.write('A3', 'To')
    worksheet.write('B3', current_year_month_day)

    # word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(long_text)
    image_path = f"{MODEL_PATH}/financial_report/{stock_code}_wordcloud.png"
    wordcloud.to_file(image_path)
    worksheet.insert_image('D3', image_path)


if __name__ == "__main__":
    user_interaction()
