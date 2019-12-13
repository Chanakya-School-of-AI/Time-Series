# This code is under develop.
# This will break.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import requests, io

def parser(x):
    return pd.datetime.strptime(x,'%Y-%m-%d')

# Fetch csv from url
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv"
s=requests.get(url).content

data = pd.read_csv(io.StringIO(s.decode('utf-8')),index_col=0,parse_dates=[0],date_parser=parser)
plt.plot(data.Year,data.Prediction)
plt.show()

#Simple Moving Average
sma=data.Prediction.rolling(window=3).mean()
print(sma)

plt.plot(data.Year,sma,label="3 month Simple Moving Average")
plt.show()

#Weighted Moving Average
#Lambda function is to compute dot product between production and weights with rolling window.
weights = [3,2,1] 
weights=np.asarray(weights)# Converting data to numpy array 
wma = data.Prediction.rolling(3).apply(lambda prediction: np.dot(prediction, weights)/weights.sum(), raw=True)
print(wma)

plt.plot(data.Year,wma,label="3 year weighted moving average")
plt.show()

#Comparision
plt.plot(data['Prediction'],label="Prediction")
plt.plot(wma, label="3 year weighted moving average")
plt.plot(sma, label="3 year Simple Moving Average")
plt.xlabel("Year")
plt.ylabel("Prediction")
plt.legend()
plt.show()
