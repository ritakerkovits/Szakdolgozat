import yfinance as yf
import pandas as pd
import datetime as dt


tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'JNJ', 'V', 'JPM', 'PG', 'WMT', 'MA', 'UNH', 'HD', 'INTC']

start_date = dt.datetime(2018,1,1)
end_date = dt.datetime(2020,12,30) 

sp500_data = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Adj Close']] 
    sp500_data = pd.concat([sp500_data, data], axis=1)
    sp500_data.rename(columns={'Adj Close': ticker}, inplace=True) 
    sp500_data = sp500_data.dropna(axis='columns') 
    
sp500_data.to_csv(r'files\\sp500_top15.csv')



