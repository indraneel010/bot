import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import matplotlib.pyplot as plt

class StatArbBot:
    def __init__(self, stock1, stock2, lookback=20, entry_z=2, exit_z=0):
        self.stock1 = stock1
        self.stock2 = stock2
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.data = None
        self.model = None

    def fetch_data(self, start_date, end_date):
        self.data = yf.download([self.stock1, self.stock2], start=start_date, end=end_date)['Adj Close']
        self.data.dropna(inplace=True)
        print("Data fetched successfully.")

    def find_cointegration(self):
        X = self.data[self.stock1]
        Y = self.data[self.stock2]
        self.model = sm.OLS(Y, sm.add_constant(X)).fit()
        p_value = sm.tsa.stattools.adfuller(self.model.resid)[1]
        print(f"P-value for cointegration: {p_value:.4f}")
        return p_value < 0.05

    def generate_signals(self):
        X = self.data[self.stock1]
        Y = self.data[self.stock2]
        self.data['Spread'] = Y - self.model.predict(sm.add_constant(X))
        self.data['ZScore'] = StandardScaler().fit_transform(self.data['Spread'].values.reshape(-1, 1))
        self.data['Signal'] = 0
        self.data.loc[self.data['ZScore'] > self.entry_z, 'Signal'] = -1  # Short spread
        self.data.loc[self.data['ZScore'] < -self.entry_z, 'Signal'] = 1  # Long spread
        self.data.loc[self.data['ZScore'].abs() < self.exit_z, 'Signal'] = 0  # Exit

    def backtest(self, capital=100000, spread_allocation=0.1):
        self.data['Portfolio'] = capital
        for i in range(1, len(self.data)):
            signal = self.data['Signal'].iloc[i - 1]
            spread_change = self.data['Spread'].iloc[i] - self.data['Spread'].iloc[i - 1]
            self.data['Portfolio'].iloc[i] = (
                self.data['Portfolio'].iloc[i - 1] +
                signal * spread_allocation * capital * spread_change
            )
        print("Backtest completed.")

    def plot_results(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.data['Portfolio'], label="Portfolio Value")
        plt.title("Statistical Arbitrage Bot Performance")
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    bot = StatArbBot('AAPL', 'MSFT', entry_z=2, exit_z=0.5)
    bot.fetch_data('2020-01-01', '2023-01-01')
    if bot.find_cointegration():
        bot.generate_signals()
        bot.backtest()
        bot.plot_results()
    else:
        print("Pairs are not cointegrated. Cannot proceed.")
