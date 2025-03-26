# Algorithm built in QuantConnect for backtesting my Reinforcement Learning using a daily Q Table

# region imports
from AlgorithmImports import *
import pandas as pd
import urllib.request
import numpy as np
from io import StringIO
# endregion

class DancingBrownHippopotamus(QCAlgorithm):

    # Initialize the data and a scheduled Backtest
    def initialize(self):
        self.set_start_date(2023, 9, 9)
        self.set_cash(100000)
        self.symbol = self.add_equity("MSFT", Resolution.Hour).Symbol
        self.csv_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQ1AjNJ75bDNi0-8tlpj7xz6YJHKPbYRg8SgUe4e-iT2BAIsYjHBNDVOtGx8gIz05MObdID9iENn52z/pub?output=csv"
        self.csv_content = self.Download(self.csv_url)
        self.df1 = self.GetStockData()
        self.q_table = self.GenQTable()
        self.ema = self.EMA(self.symbol, 2, Resolution.Daily)
        self.sma = self.SMA(self.symbol, 2, Resolution.Daily)
        self.rsi = self.RSI(self.symbol, 14, Resolution.Daily)
        self.previous_ema_values = []
        self.previous_sma_values = []
        self.previous_rsi_values = []
        self.Schedule.On(self.DateRules.EveryDay(self.symbol), self.TimeRules.Every(TimeSpan.FromHours(1)), self.OnHourlyInterval)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.BeforeMarketClose(self.symbol, 1), self.UpdateEMAHistory)
        self.holding = False
        self.selling = False

    # Retrieve EMA, SMA, and RSI data for the previous two days
    def UpdateEMAHistory(self):
        if self.ema.IsReady:
            self.previous_ema_values.append(self.ema.Current.Value)

            if len(self.previous_ema_values) > 2:
                self.previous_ema_values.pop(0)

        if self.sma.IsReady:
            self.previous_sma_values.append(self.sma.Current.Value)
            if len(self.previous_sma_values) > 2:
                self.previous_sma_values.pop(0)
        if self.rsi.IsReady:
            self.previous_rsi_values.append(self.rsi.Current.Value)
            if len(self.previous_rsi_values) > 2:
                self.previous_rsi_values.pop(0)
        self.holding = False
        

    # Function for retrieving my reinforcement learning model (the parameters from my q table)
    def GetStockData(self):
        try:
            df = pd.read_csv(StringIO(self.csv_content))
            return df
        except Exception as e:
            self.Debug(f"Error loading CSV: {str(e)}")
            return pd.DataFrame()

    # Function for parsing the excel data for the parameters from my q table into a dictionary
    def GenQTable(self):
        q_table = {}
        for i in range(len(self.df1)):
            string = str(self.df1["State"][i])
            string2 = string[:-1]
            states = string2.split(",")
            status = []
            for sta in states:
                status.append(int(sta))
            if(len(states) != 8):
                self.Debug(f"State string too long or short")
            actions =[]
            actions.append(self.df1["Sell"][i])
            actions.append(self.df1["Hold"][i])
            actions.append(self.df1["Buy"][i])
            q_table[tuple(status)] = actions
        return q_table

    # Bulk of backtesting alorithm running each hour that plugs each current state into the parameters from my q table
    # to determine buy, sell, and hold signals
    def OnHourlyInterval(self):
        
        if self.Portfolio[self.symbol].Quantity == 0:
            if self.Portfolio.TotalUnrealizedProfit < -1000:
                self.Liquidate(self.symbol)
                self.Debug(f"SELL")
        
        if len(self.previous_ema_values) < 2 or len(self.previous_sma_values) < 2 or len(self.previous_rsi_values) < 2:
            return

        # Get most recent price
        current_price = self.Securities[self.symbol].Price
        # Retrieve last 3 days of daily data
        history = self.History(self.symbol, 4, Resolution.Daily)
        if history.empty:
            return

        state = []
        state2 = []

        
        prev_day_close = history["close"].values[-2]
        day_before_prev_close = history["close"].values[-3]
        day_third = history["close"].values[-4]
        d1 = (prev_day_close - day_before_prev_close)/day_before_prev_close
        d1_2 = (day_before_prev_close - day_third)/day_third
        d2 = d1 - d1_2

        d1New = (current_price - prev_day_close)/prev_day_close
        d2New = d1New - d1

        if d1 > 0.01:
            state.append(3)
        elif d1 > 0:
            state.append(2)
        elif d1 > -0.01:
            state.append(1)
        else:
            state.append(0)
        
        if d1New > 0.01:
            state2.append(3)
        elif d1New > 0:
            state2.append(2)
        elif d1New > -0.01:
            state2.append(1)
        else:
            state2.append(0)

        
        if(d2 > 0):
            state.append(1)
        else:
            state.append(0)

        if(d2New > 0):
            state2.append(1)
        else:
            state2.append(0)

        prev_day_rsi = self.previous_rsi_values[1]

        if(prev_day_rsi > 70):
            state.append(5)
        elif prev_day_rsi > 60:
            state.append(4)
        elif prev_day_rsi > 50:
            state.append(3)
        elif prev_day_rsi > 40:
            state.append(2)
        elif prev_day_rsi > 30:
            state.append(1)
        else:
            state.append(0)

        state2.append(state[2])


        prev_day_vol = history["volume"].values[-2]
        day_before_prev_vol = history["volume"].values[-3]
        day_third_vol = history["volume"].values[-4]

        d1Vol = (prev_day_vol - day_before_prev_vol)/day_before_prev_vol
        d1Vol_2 = (day_before_prev_vol - day_third_vol)/day_third_vol
        d2Vol = d1Vol - d1Vol_2

        if d1Vol > 0.2:
            state.append(3)
        elif d1Vol > 0:
            state.append(2)
        elif d1Vol > -.02:
            state.append(1)
        else:
            state.append(0)
        
        if d2Vol > 0:
            state.append(1)
        else:
            state.append(0)

        state2.append(state[3])
        state2.append(state[4])

        prev_day_open = history["open"].values[-2]
        curr_day_open = history["open"].values[-1]
        ocNew = current_price - curr_day_open
        oc = prev_day_close - prev_day_open

        
        dema = (prev_day_close - self.previous_ema_values[1])/self.previous_ema_values[1]

        dma = (self.previous_sma_values[1] - self.previous_ema_values[1])/self.previous_ema_values[1]

        if dema > .0066:
            state.append(3)
        elif dema > 0:
            state.append(2)
        elif dema < -.0044:
            state.append(0)
        else:
            state.append(1)

        state2.append(state[5])
        
        if dma > .0033:
            state.append(3)
        elif dma > 0:
            state.append(2)
        elif dma < -.003:
            state.append(0)
        else:
            state.append(1)

        state2.append(state[6])


        if oc >0:
            state.append(1)
        else:
            state.append(0)

        if ocNew > 0:
            state2.append(1)
        else:
            state2.append(0)

        
        
        stateTuple = tuple(state)
        stateTuple2 = tuple(state2)

        if len(state) != 8:
            self.Debug("state too small")
            return
        if len(state2) != 8:
            self.Debug("state2 too small")
            return

        if (self.q_table.get(stateTuple),-20) == -20:
            self.Debug("problemo")
            return
        max2 = 0
        if (self.q_table.get(stateTuple2),-20) == -20:
            self.Debug("problemo with state 2")
        else:
            max2 = np.argmax(self.q_table.get(stateTuple2))
        max = np.argmax(self.q_table.get(stateTuple))
        if self.Portfolio[self.symbol].Quantity == 0.0 and self.holding == False:
            if max == 2:
                self.SetHoldings(self.symbol, 1)
                self.Debug(f"BUY Signal: {self.Time} State: {stateTuple} Price: {current_price} oldPrice: {prev_day_close}")
                self.holding = True
                self.selling = False
                return
        elif self.Portfolio[self.symbol].Quantity > 0.0 and self.selling == False:
            if max == 0:
                value = self.Portfolio[self.symbol].Quantity
                self.Liquidate(self.symbol)
                self.Debug(f"SELL Signal: {self.Time} State: {stateTuple} Price: {current_price}")
                self.selling = True
                return
        return



    def on_data(self, data: Slice):
        pass
        

        
