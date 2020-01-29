import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from pandas import datetime
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import LSTM,GRU

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

data=np.array(scaled)

X = []
y = []
for i in range(1000, len(scaled)):
    X.append(data[i-1000:i, 0:])#0:for all col
    y.append(data[i, 0:])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1],6))

###LSTM###

regressor_lstm = Sequential()
regressor_lstm.add(LSTM(units = 128, return_sequences = True, input_shape = (X.shape[1], 6)))
regressor_lstm.add(LSTM(units = 128, return_sequences = True))
regressor_lstm.add(LSTM(units = 128, return_sequences = True))
regressor_lstm.add(LSTM(units = 128))
regressor_lstm.add(Dense(units = 6))
regressor_lstm.compile(loss='mse', optimizer='adam',metrics=['accuracy'])
regressor_lstm.summary()

#Fitting the model
history_lstm = regressor_lstm.fit(X, y, epochs=5, batch_size=800, validation_split=0.3,  shuffle=False)

###GRU###

regressor_gru = Sequential()

regressor_gru.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], 5)))
regressor_gru.add(Dropout(0.2))

regressor_gru.add(LSTM(units = 50, return_sequences = True))
regressor_gru.add(Dropout(0.2))

regressor_gru.add(LSTM(units = 50, return_sequences = True))
regressor_gru.add(Dropout(0.2))

regressor_gru.add(LSTM(units = 50))
regressor_gru.add(Dropout(0.2))

regressor_gru.add(Dense(units = 5))

regressor_gru.compile(loss='mse', optimizer='adam',metrics=['accuracy'])

regressor_gru.summary()

#Fitting the model
history_gru = regressor_gru.fit(X, y, epochs=100, batch_size=40, validation_split=0.25,  shuffle=False)

#plotting the loss
plt.plot(history_lstm.history['accuracy'],label='LSTM train',color='red')
plt.plot(history_lstm.history['val_accuracy'],label='LSTM test',color='blue')
plt.plot(history_gru.history['accuracy'],label='LSTM train',color='green')
plt.plot(history_gru.history['val_accuracy'],label='LSTM test',color='black')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
