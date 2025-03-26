# File for uploading intraday stock data to excel along with technical indicators using Polygon Stocks API instead of
# yfinance. Since PolygonStocks has API rate limits and only outputs a small maximum of data points for intraday data
# I had to iteratively call the API to output a lot of data

import requests
import pandas as pd
from datetime import datetime, timedelta

# Polygon API parameters
api_key = 'dfOEh9yfeSaVSJH2lrGDvlsEYbKZ3IeL'
symbol = 'AAPL'
multiplier = 1      # 4-hour bars
timespan = 'hour'
#from_date = '2023-05-01'
#to_date = '2023-05-08'
from_date = '2023-12-09'
to_date = '2023-12-16'
date_format = "%Y-%m-%d"
numDaysFrom = 8
numDaysTo = 7

datetime_start = datetime.strptime(from_date, date_format)

# Function for changing my data originally set to hourly data to quarterly data
def change():
    global from_date
    global to_date
    global multiplier
    global symbol
    global numDaysFrom
    global numDaysTo
    global datetime_start
    from_date = '2024-12-24'
    to_date = '2025-01-03'
    multiplier = 4
    symbol = 'GOOG'
    numDaysFrom = 11
    numDaysTo = 10
    datetime_start = datetime.strptime(from_date, date_format)


# Function for retrieving data
def getData(symbol,multiplier,timespan,start, end, api_key):
    # API URL
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "apiKey": api_key
    }

    # Fetch data
    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    if 'results' in data:
        df = pd.DataFrame(data['results'])
        df['t'] = pd.to_datetime(df['t'], unit='ms')  # Convert timestamp to datetime
        df.rename(columns={'t': 'timestamp', 'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
        return df
    else:
        print("No data found:", data)

    data2 = [['timestamp', 0], ['open', 1], ['high',2],['low',3],['close', 4],['volume',5]]

    return pd.DataFrame(data2)

#df1 = getData(symbol, multiplier, timespan, from_date, to_date, api_key)
change()
df1 = pd.read_excel("StockData4HR.xlsx")

# Function for iteratively calling getData with increasing dates to bypass Polygon's low API limits
for i in range(4):
    datetime_start = datetime_start + timedelta(days = numDaysFrom)
    print(datetime_start)
    datetime_end = datetime_start + timedelta(days = numDaysTo)
    from_date = datetime_start.strftime("%Y-%m-%d")
    print(from_date)
    to_date = datetime_end.strftime("%Y-%m-%d")
    print(to_date)
    df2 = getData(symbol, multiplier, timespan, from_date, to_date, api_key)
    df1 = pd.concat([df1, df2])

#df1 = getData(symbol, multiplier, timespan, from_date, to_date, api_key)

# Outputting data to excel
output_file = "StockData4HR.xlsx"
df1.to_excel(output_file, index=False)
print(f"Data saved")
