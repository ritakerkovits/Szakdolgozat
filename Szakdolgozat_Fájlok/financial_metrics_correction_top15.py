import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np



#--------------
#FUNCTIONS
#--------------


def read_weights(path, weight_column = "Weights"):
    csv_file = pd.read_csv(path)
    return pd.Series(csv_file[weight_column].values, index=csv_file.Tickers).to_dict()

def download_data(tickers_dict, start_date, end_date):
    df = pd.DataFrame()
    for ticker in tickers_dict.keys():
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data[['Adj Close']]
        df = pd.concat([df, data], axis=1)
        df.rename(columns={'Adj Close': ticker}, inplace=True)
    df = df.dropna(axis='columns')
    return df

def calculate_weighted_log_returns(df, weights_dict):
    log_returns = np.log(df / df.shift(1)).dropna()
    for ticker in weights_dict:
        log_returns[ticker] = log_returns[ticker] * weights_dict[ticker]
    row_sum = log_returns.sum(axis = 1)
    return row_sum

def calculate_financial_metrics(column_name):
    annual = df_combined[column_name].sum()
    avg_return = df_combined[column_name].mean()
    volatility = df_combined[column_name].std()
    sharpe_ratio = (annual - risk_free_rate) / volatility
    df_result.at['Annual Return', column_name] = annual
    df_result.at['Average Return', column_name] = avg_return
    df_result.at['Volatility', column_name] = volatility
    df_result.at['Sharpe Ratio', column_name] = sharpe_ratio / 100
    return df_result
    


#--------------
#MAIN PROCESS
#--------------
start_date = dt.datetime(2021,1,1)
end_date = dt.datetime(2021,12,31) 

columns = ['Markowitz', 'Sharpe', 'VADER Positive', 'VADER Compound', 'TextBlob']
index_labels = ['Annual Return', 'Average Return', 'Volatility', 'Sharpe Ratio']
df_result = pd.DataFrame(columns=columns, index=index_labels)
risk_free_rate = 0.02

# tickerek és a súlyok szótárba olvasása a további számításokhoz
dict_markowitz = read_weights(r'files\\Markowitz_weights_top15.csv')
dict_sharpe = read_weights(r'files\\Sharpe_weights_top15.csv')
dict_sentiment_pos = read_weights(r'files\\Sentiment_weights.csv', 'Weights VADER Positive')
dict_sentiment = read_weights(r'files\\Sentiment_weights.csv', 'Weights VADER Compound')
dict_textblob = read_weights(r'files\\Sentiment_weights.csv', 'Weights TextBlob Polarity')


# a tickerek alapján letölti a yahoo oldaláról az adj close adatokat 2021-es évre
df_markowitz = download_data(dict_markowitz, start_date, end_date)
df_markowitz.to_csv(r'files\\Markowitz_2021_top15.csv')
df_sharpe = download_data(dict_sharpe, start_date, end_date)
df_sharpe.to_csv(r'files\\Sharpe_2021_top15.csv')
df_sentiment = download_data(dict_sentiment, start_date, end_date)
df_sentiment.to_csv(r'files\\Sentiment_2021_top15.csv')


# Calculating weighted log returns
# eszközökre számolt napi súlyozott loghozamok, majd ezek összeadása egy napra (adott dátumra)
# egyes portfoliókra számolt napi súlyozott loghozamok - oszlop a portfoliócsomag, sorok a napi súlyozott portfolió loghozam
df_markowitz = calculate_weighted_log_returns(df_markowitz, dict_markowitz)
df_sharpe = calculate_weighted_log_returns(df_sharpe, dict_sharpe)
df_sentiment_compound = calculate_weighted_log_returns(df_sentiment, dict_sentiment)
df_sentiment_positive = calculate_weighted_log_returns(df_sentiment, dict_sentiment_pos)
df_sentiment_textblob = calculate_weighted_log_returns(df_sentiment, dict_textblob)

df_combined = pd.concat([df_markowitz, df_sharpe, df_sentiment_compound, df_sentiment_positive, df_sentiment_textblob], axis = 1)
df_combined.columns = ['Markowitz', 'Sharpe', 'VADER Compound', 'VADER Positive', 'TextBlob']
df_combined.to_csv(r'files\\Weighted_Portfolios_top15.csv')

calculate_financial_metrics('Markowitz')
calculate_financial_metrics('Sharpe')
calculate_financial_metrics('VADER Compound')
calculate_financial_metrics('VADER Positive')
calculate_financial_metrics('TextBlob')


df_result.to_csv(r'files\\Final_Comparison_top15.csv')


