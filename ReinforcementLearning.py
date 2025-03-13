import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

action_count = 3
q_table = {}

alpha = 0.3
gamma = 0.1
epsilon = 0.1
df = pd.read_excel("/Users/gabrielhaskell/Documents/Personal/StockData3.xlsx")
df3 = pd.read_excel("/Users/gabrielhaskell/Documents/Personal/Stockx.xlsx")
df4 = pd.read_excel("/Users/gabrielhaskell/Documents/Personal/PolygonStocks.xlsx")
df5 = pd.read_excel("/Users/gabrielhaskell/Documents/Personal/StockDataMSFT.xlsx")
#Can have prev_action be apart of state and if in buy state can decide whether to buy more or hold or sell
    

    
def getState(n,df1):
    state = []
    state.append(df1["RLMD1"][n])
    state.append(df1["RLD2REAL"][n])
    state.append(df1["RLMRSI"][n])
    state.append(df1["RLMD1Volume"][n])
    state.append(df1["RLD2Volume"][n])
    state.append(df1["RLMDEMA"][n])
    state.append(df1["RLMDMA"][n])
    state.append(df1["RLOC"][n])
    return tuple(state)

def calcReward(action, n, df1):
    price_init = df1["Close"][n]
    price_next = df1["Close"][n+1]
    price_next2 = df1["Close"][n+2]
    profit = price_next - price_init
    profit = profit / (price_init)
    profit = profit * 100
    profit2 = price_next2 - price_init
    profit2 = profit2 / (price_init)
    profit2 = profit2 * 100
    reward = 0
    if(action == 2):
        if(profit > 0):
            #reward = 14 * (profit/13.28794098)
            reward = 8.6 * profit
        if(profit < 0):
            #reward = 7 * (((profit + 11.2682)/11.2682) -1)
            reward = 4.4 * profit
        # if(profit2 >0):
        #     reward = reward + profit2
        # if(profit2 < 0):
        #     reward = reward + 0.7 * profit2
        return reward
    if(action == 1):
        if(profit > .33):
            #reward = (-7.0) * (profit/13.28794098)
            reward = (-1.7) * profit
            return reward
        if(profit > 0 and profit <= 0.33):
            reward = 3.2 * profit
        if(profit < 0):
            #reward = (-2.0) * ((((profit + 11.2682)/11.2682) -1))
            reward = (-4.0) * profit
            return reward
    if(action == 0):
        if(profit > 0):
            #reward = (-8.0) * (profit/13.28794098)
            reward = (-2.6) * profit
        if(profit < 0):
            #reward = (-10.0) * ((((profit + 11.2682)/11.2682) -1))
            reward = (-9.2) * profit
        # if(profit2 < 0):
        #     reward = reward - profit2
        # if(profit2 > 0):
        #     reward = reward - 0.5 * profit2
        return reward
    return reward






def train():
    for i in range(2000):
        reward = 0
        global alpha
        global gamma
        global epsilon
        if(alpha > 0.1):
            alpha -= .0005
        for x in range(len(df)-4):
            state = getState(x,df)
            q_table.setdefault(state, [0 for _ in range(action_count)])
            if random.uniform(0,1) < epsilon:
                action = random.randint(0,2)
            else:
                action = np.argmax(q_table.get(state))
            next_state = getState(x+1,df)
            reward = calcReward(action, x, df)
            prev_val = q_table.get(state)[action]
            q_table.setdefault(next_state, [0 for _ in range(action_count)])
            next_max = np.max(q_table.get(next_state))

            new_val = (1 - alpha) * prev_val + alpha * (reward + gamma * next_max)

            q_table.get(state)[action] = new_val

        for x in range(len(df3)-4):
            state = getState(x, df3)
            q_table.setdefault(state, [0 for _ in range(action_count)])
            if random.uniform(0,1) < epsilon:
                action = random.randint(0,2)
            else:
                action = np.argmax(q_table.get(state))
            next_state = getState(x+1, df3)
            reward = calcReward(action, x, df3)
            prev_val = q_table.get(state)[action]
            q_table.setdefault(next_state, [0 for _ in range(action_count)])
            next_max = np.max(q_table.get(next_state))

            new_val = (1 - alpha) * prev_val + alpha * (reward + gamma * next_max)

            q_table.get(state)[action] = new_val
        
        for x in range(len(df5)-4):
            state = getState(x, df5)
            q_table.setdefault(state, [0 for _ in range(action_count)])
            if random.uniform(0,1) < epsilon:
                action = random.randint(0,2)
            else:
                action = np.argmax(q_table.get(state))
            next_state = getState(x+1, df5)
            reward = calcReward(action, x, df5)
            prev_val = q_table.get(state)[action]
            q_table.setdefault(next_state, [0 for _ in range(action_count)])
            next_max = np.max(q_table.get(next_state))

            new_val = (1 - alpha) * prev_val + alpha * (reward + gamma * next_max)

            q_table.get(state)[action] = new_val


train()

dataframe = {'State': [], 'Sell': [], 'Hold': [], 'Buy':[]}

for key in q_table:
    string = ""
    for num in key:
        string += str(num) + ","
    dataframe.get("State").append(string)
    dataframe.get("Sell").append(q_table.get(key)[0])
    dataframe.get("Hold").append(q_table.get(key)[1])
    dataframe.get("Buy").append(q_table.get(key)[2])

df2 = pd.DataFrame(dataframe)
df2.to_excel("PredictedStocks.xlsx", index = False)
print("q_table written to excel")