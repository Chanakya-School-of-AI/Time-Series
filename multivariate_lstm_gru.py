import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import sqrt
from numpy import concatenate
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from pandas import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Dropout

def parser(x):
    return datetime.strptime(x,'%d-%m-%Y %H:%M')

dataset=pd.read_csv(r"D:\occupancy_data\multitime.csv",index_col=0,parse_dates=[0],date_parser=parser)
dataset.head(10)

values=np.array(dataset)

#Normalizing input features
scaler=MinMaxScaler(feature_range=(0,1))
scaled=scaler.fit_transform(values)

scaled=pd.DataFrame(scaled)
scaled.head(10)

train=scaled[:int(0.8*(len(scaled)))]
train.shape
train=np.array(train)

valid=scaled[int(0.8*(len(scaled))):]
valid.shape
valid=np.array(valid)

X_train = []
y_train = []
for i in range(200, len(train)):
    X_train.append(train[i-200:i, 0])
    y_train.append(train[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = []
y_test = []
for i in range(200, len(valid)):
    X_test.append(valid[i-200:i, 0])
    y_test.append(valid[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

###LSTM###
regressor_lstm = Sequential()

regressor_lstm.add(LSTM(50, return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
regressor_lstm.add(LSTM(units=20, return_sequences=True))
regressor_lstm.add(LSTM(units=20))
regressor_lstm.add(Dense(units=1))

regressor_lstm.compile(loss='mae', optimizer='adam')

regressor_lstm.summary()

#Fitting the model
history_lstm = regressor_lstm.fit(X_train, y_train, epochs=1, batch_size=50, validation_data=(X_test,y_test),  shuffle=False)

###GRU###
regressor_gru=Sequential()

regressor_gru.add(GRU(50, return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])))
regressor_gru.add(GRU(units=20, return_sequences=True))
regressor_gru.add(GRU(units=20))
regressor_gru.add(Dense(units=1))

regressor_gru.compile(loss='mae', optimizer='adam')

regressor_gru.summary()

#Fitting the model
history_gru = regressor_gru.fit(X_train, y_train, epochs=1, batch_size=50, validation_data=(X_test,y_test),  shuffle=False)

#plotting the loss
plt.figure(figsize=(10,6),dpi=100)
plt.plot(history_lstm.history['loss'], label='LSTM train', color='red')
plt.plot(history_lstm.history['val_loss'], label='LSTM test', color= 'green')
plt.plot(history_gru.history['loss'], label='GRU train', color='brown')
plt.plot(history_gru.history['val_loss'], label='GRU test', color='blue')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.title('training and validation loss')
plt.show()
