import yfinance as yf
import pandas as pd
import datetime as dt

df = pd.read_csv(r'files\\new_constituents.csv')
tickers = df['Ticker'].tolist()

start_date = dt.datetime(2018,1,1)
end_date = dt.datetime(2020,12,30) 

sp500_data = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Adj Close']] 
    sp500_data = pd.concat([sp500_data, data], axis=1)
    sp500_data.rename(columns={'Adj Close': ticker}, inplace=True) 
    sp500_data = sp500_data.dropna(axis='columns') 
    
sp500_data.to_csv(r'files\\sp500_prices_new.csv')



