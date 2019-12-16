import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import train_test_split

from pmdarima.arima import auto_arima
# Pmdarima operates by wrapping statsmodels.tsa.ARIMA and statsmodels.tsa.statespace.SARIMAX 
# into one estimator class and creating a more user-friendly estimator interface 
# for programmers familiar with scikit-learn.

import requests, io

def parser(x):
    return pd.datetime.strptime(x,'%Y-%m-%d')

print("\n----------------- CSOAI's ARIMA tutorial ----------------- \n")
print("Keep reading the comments in the code for more context")
print("hit enter key to proceed...")
input()

# Fetch csv from url
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv"
s=requests.get(url).content

# Indexing of dates is neccessary for times series anlysis
birth = pd.read_csv(io.StringIO(s.decode('utf-8')), index_col=0, parse_dates=[0], date_parser=parser)

print("\n----------------- Fitting ARIMA automatically ----------------- \n")
# Auto Model
# We are using auto_arima to automatically find the right orders -> p,d,q
model = auto_arima(birth, trace=True, error_action='ignore', suppress_warnings=True,stationary=True)
model.fit(birth)
print("Notice the ARIMA order and their AIC and BIC values.")

print("\n----------------- Fitting ARIMA manually ----------------- \n")
print("hit enter to proceed...")
input()
# Manual Model

birth.head()
birth.plot()
# This plot will demo that the data is stationary because we don't have any trend
plt.show()

print("\nThe graph doesn't show any trend.")
# We will still do a test to confirm if data is stationary.
# Stationary: mean, var and covar is constant over period
# Augmented Dickey fuller test to test wether data is stationary or not
result = adfuller(birth.Births)

# ADF is less than 1% value. 
print('ADF Statistic: %f' % result[0])
# p-value < 0.05, hence we reject the null hypothesis that data has unit root
# It means data is stationary.
# Hence d = 0
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# Now we need to find the orders of AR and MA to pass to ARIMA model.
# PACF and ACF helps here.
plot_acf(birth)
# Since 2 lag points above the p-value zone so q = 2
plot_pacf(birth)
# Since 2 lag points above the p-value zone so p = 2
plt.show()

series = pd.read_csv(io.StringIO(s.decode('utf-8')), index_col=0, parse_dates=[0], squeeze=True, date_parser=parser)
model = ARIMA(series, order=(2,0,2))
model_fit = model.fit()
print(model_fit.summary())

#start_index = "1959-01-01"
#end_index = "1959-12-31"
#forecast = model_fit.predict(start=start_index, end=end_index)
#forecast.plot()
#plt.show()

# we can see in model summary that the AIC and BIC errors are quite high
# so lets be more conservative and chose the lags which are significantly above the p-value zone.
# now the orders become p = 1 and q = 1 with d = 0.
model = ARIMA(series, order=(1,0,1))
model_fit = model.fit()
print(model_fit.summary())

#forecast = model_fit.predict(start=start_index, end=end_index)
#forecast.plot()
#plt.show()

