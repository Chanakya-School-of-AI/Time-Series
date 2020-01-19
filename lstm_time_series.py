import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pandas import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

data=pd.read_csv(r"D:\occupancy_data\multitime.csv",index_col=0,parse_dates=[0],date_parser=parser)
data

sns.pairplot(data[data.columns[0:]])
plt.show()

#LSTM for Temperature

temperature = data[data.columns[0:1]]
temperature

train=temperature[:int(0.8*(len(temperature)))]
train.shape
train=np.array(train)

valid=temperature[int(0.8*(len(temperature))):]
valid.shape

sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(train)

X_train = []
y_train = []
for i in range(200, len(train)):
    X_train.append(training_set_scaled[i-200:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

regressor = Sequential()
#Adding a first layer
regressor.add(LSTM(units = 20, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second layer
regressor.add(LSTM(units = 20, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third layer
regressor.add(LSTM(units = 20, return_sequences = True))
regressor.add(Dropout(0.2))

#Adding a fourth layer
regressor.add(LSTM(units = 20))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Using adam optimizer
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#fitting RNN to the training set
regressor.fit(X_train, y_train)

# Getting the prediction on test
inputs = valid[len(temperature) - len(valid) - 200:].values
inputs = inputs.reshape(-1,1)
inputs = sc.fit_transform(valid)
X_test = []
for i in range(200, len(valid)):
    X_test.append(inputs[i-200:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_test = regressor.predict(X_test)
predicted_test = sc.inverse_transform(predicted_test)

#Error in the plots codes
plt.plot(valid, color = 'red', label = 'Real values')
plt.show()
plt.plot(predicted_test, color = 'blue', label = 'Predicted values')
plt.show()
