import pandas as pd
import numpy as np

from matplotlib import pyplot
from pandas import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from numpy import concatenate

def parser(x):
    return datetime.strptime(x,'%d-%m-%Y %H:%M')

data=pd.read_csv(r"D:\occupancy_data\multitime.csv",index_col=0,parse_dates=[0],date_parser=parser)
data

values=data.values

groups=[0,1,2,3,4]
i=1

pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups),1,i)
    pyplot.plot(values[:,group])
    pyplot.title(data.columns[group],y=0.5,loc='right')
    i+=1
pyplot.show()

train=data[:int(0.8*(len(data)))]
train.shape
train=np.array(train)

valid=data[int(0.8*(len(data))):]
valid.shape

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

X_train = []
y_train = []
for i in range(200, len(train)):
    X_train.append(scaled[i-200:i, 0])
    y_train.append(scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


inputs = valid[len(data) - len(valid) - 200:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.fit_transform(valid)
X_test = []
y_test=[]
for i in range(200, len(valid)):
    X_test.append(inputs[i-200:i, 0])
    y_test.append(inputs[i,0])
X_test,y_test = np.array(X_test),np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

regressor = Sequential()
regressor.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
regressor.add(Dense(1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

history = regressor.fit(X_train,y_train, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# Again error in plots (plot history)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# make a prediction
yhat = regressor.predict(X_test)
