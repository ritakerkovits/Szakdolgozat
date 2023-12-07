import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models


#--------------
#FUNCTIONS
#--------------

def optimize_portfolio(log_returns, cov_matrix, optimize_function):
    ef = EfficientFrontier(log_returns, cov_matrix)
    optimize_function(ef)
    clean_weights = ef.clean_weights()
    print(ef.portfolio_performance(verbose=True))
    return {ticker: round(weight, 4) for ticker, weight in clean_weights.items() if weight != 0}

def write_to_csv(filename, data_dict):
    with open(filename, 'w') as file:
        file.write('Tickers,Weights\n')
        for key, value in data_dict.items():
            file.write(f'{key},{value}\n')


#--------------
#MAIN PROCESS
#--------------

df = pd.read_csv(r'files\\sp500_top15.csv', parse_dates=True)
df.set_index('Date', inplace=True)
log_returns = np.log(df / df.shift(1)).dropna()
annualized_log_returns = log_returns.mean() * 252
cov_matrix = risk_models.risk_matrix(df, method='sample_cov', log_returns=True)

assets_sharpe = optimize_portfolio(annualized_log_returns, cov_matrix, lambda ef: ef.max_sharpe())
assets_min = optimize_portfolio(annualized_log_returns, cov_matrix, lambda ef: ef.min_volatility())

write_to_csv(r"files\\Sharpe_weights_top15.csv", assets_sharpe)
write_to_csv(r"files\\Markowitz_weights_top15.csv", assets_min)

