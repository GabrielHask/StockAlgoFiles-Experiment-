import yfinance as yf
import ta
import pandas as pd
import datetime as dt

# Step 1: Fetch Apple stock data
ticker = "AAPL"
data = yf.download(ticker, start="2023-05-01", end="2025-01-05", progress=False)

# Step 2: Ensure all data columns are numeric
#data = data.apply(pd.to_numeric, errors="coerce")  # Convert non-numeric to NaN
#data.dropna(inplace=True)  # Drop rows with NaN values



#data = pd.read_excel("StockData4HR.xlsx")
close_series = data["Close"].squeeze()

# Step 3: Calculate Technical Indicators using `ta`
data["SMA_2"] = ta.trend.sma_indicator(close_series, window=2)
data["SMA_10"] = ta.trend.sma_indicator(close_series, window=10)
data["EMA_2"] = ta.trend.ema_indicator(close_series, window=2)
data['EMA_5'] = ta.trend.ema_indicator(close_series, window=5)
data["SMA_3"] = ta.trend.sma_indicator(close_series, window=3)
data["RSI"] = ta.momentum.rsi(close_series, window=14)

# data["MACD"] = ta.trend.macd(close_series)
# data["MACD_Signal"] = ta.trend.macd_signal(close_series)


# data["BB_upper"], data["BB_middle"], data["BB_lower"] = ta.volatility.BollingerBands(close_series, window=20, window_dev=2).bollinger_hband(), ta.volatility.BollingerBands(close_series, window=20).bollinger_mavg(), ta.volatility.BollingerBands(close_series, window=20, window_dev=2).bollinger_lband()

# VWAP (Volume Weighted Average Price)
# data["VWAP"] = ta.volume.VolumeWeightedAveragePrice(high=data["High"].squeeze(), low=data["Low"].squeeze(), close=data["Close"].squeeze(), volume=data["Volume"].squeeze()).volume_weighted_average_price()

# Step 4: Save to Excel
output_file = "StockDataAAPL.xlsx"

data.to_excel(output_file, index=True)

print(f"Stock data and indicators have been saved to '{output_file}'")
