import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, io

from pandas import datetime
from pandas import Series
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

%matplotlib inline

def parser(x):
    return pd.datetime.strptime(x,'%d-%m-%Y')

##Differencing function
def difference(y, interval=1):
    diff = list()
    for i in range(interval, len(y)):
        value = y[i]-y[i-interval]
        diff.append(value)
    return Series(diff)

##Dataset
url="http://employment_in_public_and_organised_private_sector.csv"
s=requests.get(url).content
###Got an error while uploading data from github

data=pd.read_csv(r"D:\Time Series\employment_in_public_and_organised_private_sector.csv",index_col=0, parse_dates=[0], date_parser=parser)
data.head()

##Trimed the dataset in 3 parts

###Public###
public=data[data.columns[0:1]]
public

plt.plot(public)
plt.show()

result_pub1=adfuller(public.Public)
print('ADF Statistic: %f' % result_pub1[0])
print('p-value: %f' % result_pub1[1])

public_diff1=difference(public.Public)
plt.plot(public_diff1)
plt.show()

result_pub2=adfuller(public_diff1)
print('ADF Statistic: %f' % result_pub2[0])
print('p-value: %f' % result_pub2[1])

public_diff2=difference(public_diff1)
plt.plot(public_diff2)
plt.show()

result_pub3=adfuller(public_diff2)
print('ADF Statistic: %f' % result_pub3[0])
print('p-value: %f' % result_pub3[1])

plot_acf(public_diff2)
plot_pacf(public_diff2)

model =ARIMA(public_diff2, order=(1,0,1))
model_fit = model.fit()
print(model_fit.summary())

###Private###
private=data[data.columns[1:2]]
private

plt.plot(private)
plt.show()

result_pri1=adfuller(private.Private)
print('ADF Statistic: %f' % result_pri1[0])
print('p-value: %f' % result_pri1[1])

private_diff1=difference(private.Private)
plt.plot(private_diff1)
plt.show()

result_pri2=adfuller(private_diff1)
print('ADF Statistic: %f' % result_pri2[0])
print('p-value: %f' % result_pri2[1])

private_diff2=difference(private_diff1)
plt.plot(private_diff2)
plt.show()

result_pri3=adfuller(private_diff2)
print('ADF Statistic: %f' % result_pri3[0])
print('p-value: %f' % result_pri3[1])

plot_acf(private_diff2)
plot_pacf(private_diff2)

model2 = ARIMA(private_diff2, order=(1,0,1))
model_fit2 = model2.fit()
print(model_fit2.summary())

###Number###
number=data[data.columns[2:3]]
number

plt.plot(number)
plt.show()

result_num1=adfuller(number.Number)
print('ADF Statistic: %f' % result_num1[0])
print('p-value: %f' % result_num1[1])

number_diff1=difference(number.Number)
plt.plot(private_diff1)
plt.show()

result_num2=adfuller(number_diff1)
print('ADF Statistic: %f' % result_num2[0])
print('p-value: %f' % result_num2[1])

plot_acf(number_diff1)
plot_pacf(number_diff1)

model3 = ARIMA(number_diff1, order=(1,0,1))
model_fit3 = model3.fit()
print(model_fit3.summary())
