import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
import plotext as plt
import terminalplot as tlp
import matplotlib.pyplot as pplt


def getPredictionBin(df, features_col, target, Multi, Prob, targetNames, size=0.25, randState=16):
    X = df[features_col]
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = size, random_state = randState )
    if(Multi):
        logreg = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", max_iter=200)
    else:
        logreg = LogisticRegression(random_state=16)
    logreg.fit(X_train, y_train)
    if(Prob):
        y_pred = logreg.predict_proba(X_test)
        return y_pred
    else:
        y_pred = logreg.predict(X_test)
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
        cnf_matrix
        target_names = targetNames
        print(classification_report(y_test, y_pred, target_names=target_names))
        return y_pred
    
def printPrediction(y_pred, Prob):
    if(Prob):
        for i in range(len(y_pred[0])):
            print(y_pred[:,i])
    else:
        print(y_pred)

def addToExcel(filePath, columnName, y_pred, Prob):
    df = pd.read_excel(filePath)
    if(Prob):
        for i in range(len(y_pred[0])):
            df[f"{columnName}_{i}"] = y_pred[:,i]
    else:
        df[f"{columnName}"] = y_pred
    df.to_excel(filePath, index = False)



# load dataset
df = pd.read_excel("Stockx.xlsx", header=0)

#print(df.head())

#split dataset in features and target variable
feature_cols = ['RSI', 'NDEMA_10', #'NDEMA_50',
                 'NDSMA','NDMACD','NDEMA_20', 'NDVWAP','NDBB_UPPER', 'LASTSIGS']
feature = ["NDMA"]
y = df.BUYBIN # Target variable
y2 = df.BuyProb


targets = ["Sell", "Buy"]
targets2 = ["Sell", "Moderate Sell", "Moderate Buy", "Buy"]

# y_pred = getPredictionBin(df, feature, y, False, False, targets)
# y_pred3 = getPredictionBin(df, feature_cols, y, False, True, targets)
# y_pred2 = getPredictionBin(df, feature_cols, y2, True, False, targets2)
# y_pred4 = getPredictionBin(df, feature_cols, y2, True, True, targets2)

# addToExcel("PredictedStocks.xlsx", "BinaryPredictions", y_pred, False)
# addToExcel("PredictedStocks.xlsx", "BinaryProbPredictions", y_pred3, True)
# addToExcel("PredictedStocks.xlsx", "MultiPredictions", y_pred2, False)
# addToExcel("PredictedStocks.xlsx", "MultiProbPredictions", y_pred4, True)

#y_pred = getPredictionBin(df, feature_cols, y, False, False, targets)
#addToExcel("PredictedStocks.xlsx", "BinaryPreds4", y_pred, False)

#plt.date_form('Y-m-d H:m:s')
# df2 = pd.read_excel("StockData3.xlsx", header=0)
# df3 = pd.read_excel("PredictedStocks.xlsx", header=0)

# data = df2['Close']
# preds = df3['BinaryPreds4']
# dates = df['Price']
# data2 = df['Close'][:len(df) // 5]

# data = df3["Close"]

# plt.plot(data)
# for i in range(len(data)):
#     if(df3["BinaryPredictions"][i] > 0):
#         plt.plot([i+1], [data[i]], color = 'green')


data = df['Close'][:len(df)//20]
min = df['RelMin'][:len(df)//20]

# CSma = df['CSMA']
# rsi = df['RSI']
# prof = df['Nprofit'][:len(df)//20]
# vol = df["Volume"]
# ema = df["EMA_10"][:len(df)//20]
# ema2 = df["EMA_2"][:len(df)//20]
# ema3 = df["EMA_5"][:len(df)//10]
# sma2 = df["SMA_2"][:len(df)//20]
# sma3 = df["SMA_10"][:len(df)//20]
# d1 = df["D1"][:len(df)//10]
# d2 = df["D2"][:len(df)//10]
# zero = df["Z"][:len(df)//20]
# csma = df["CMA"][:len(df)//10]
# cema = df["CEMA"][:len(df)//10]
# dma = df["DMA"][:len(df)//20]

plt.plot(data)
# plt.plot(dma)
# plt.plot(zero)
# plt.plot(data2)
#plt.plot(sma2, color = "red")
#plt.plot(ema2, color = "green")
#plt.plot(d1)
#plt.plot(d2, color = "red")
#plt.plot(zero)
for i in range(len(data)):
    if((min[i]>0)):
        plt.plot([i],[data[i]], color = 'green')
    if((min[i]<0)):
        plt.plot([i+1],[data[i]], color = "red")




# plt.title("Google Stock Price CandleSticks")
# plt.xlabel("Date")
# plt.ylabel("Stock Price $")



plt.show()







# split X and y into training and testing sets
# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16)

# X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y2, test_size=0.25, random_state=16)

# # import the class
# from sklearn.linear_model import LogisticRegression

# # instantiate the model (using the default parameters)
# logreg = LogisticRegression(random_state=16)
# multilogreg = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", max_iter=200)
# #print("hello", flush = True)
# # fit the model with data
# logreg.fit(X_train, y_train)
# multilogreg.fit(X_train2, y_train2)

# # print("hello", flush = True)
# y_pred = logreg.predict_proba(X_test)
# y_pred2 = logreg.predict(X_test)
# y_predMulti = multilogreg.predict_proba(X_test2)
# y_predMulti2 = multilogreg.predict(X_test2)

# print(y_predMulti2)

# from sklearn import metrics

# cnf_matrix = metrics.confusion_matrix(y_test, y_pred2)
# cnf_matrix
# from sklearn.metrics import classification_report
# target_names = ['sell', 'buy']
# print(classification_report(y_test, y_pred2, target_names=target_names))

# cnf_matrix = metrics.confusion_matrix(y_test2, y_predMulti2)
# cnf_matrix
# from sklearn.metrics import classification_report
# target_names = ['sell', 'moderate', 'buy']
# print(classification_report(y_test2, y_predMulti2, target_names=target_names))




# # print(y_pred, flush = True)
# # print(y_predMulti[:,1], flush = True)

# # p = y_pred[:, 1]
# #print("Probability of class 1:\n", p)

# # df = pd.read_excel("PredictedStocks.xlsx")
# # df["PredictedProb"] = p
# # df["PredictedBin"] = y_pred2

# # df.to_excel("PredictedStocks.xlsx", index = False)
# # df.to_excel("PredictedStocks.xlsx", index=False)

# # for i in range(3):
# #     df[f"PredictedProbClass_{i}"] = y_predMulti[:,i]
# #     df.to_excel("PredictedStocks.xlsx", index = False)

# # import the metrics class
# #from sklearn import metrics

# # cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
# # cnf_matrix

# # # array([[115,   8],
# # #       [ 30,  39]])


# # # # import required modules
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # import seaborn as sns

# # # class_names=[0,1] # name  of classes
# # # fig, ax = plt.subplots()
# # # tick_marks = np.arange(len(class_names))
# # # plt.xticks(tick_marks, class_names)
# # # plt.yticks(tick_marks, class_names)
# # # # create heatmap
# # # sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
# # # ax.xaxis.set_label_position("top")
# # # plt.tight_layout()
# # # plt.title('Confusion matrix', y=1.1)
# # # plt.ylabel('Actual label')
# # # plt.xlabel('Predicted label')

# # # Text(0.5,257.44,'Predicted label')



