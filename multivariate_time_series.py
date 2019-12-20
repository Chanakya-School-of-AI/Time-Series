import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from pandas import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
%matplotlib inline

#function
def parser(x):
    return datetime.strptime(x,'%d-%m-%Y %H:%M')

#Read the data
mts=pd.read_csv(r'D:\Time Series\occupancy_data\multivariate_time_series_dataset.csv',index_col=0,parse_dates=[0],date_parser=parser)
mts.head()

#Check the data type
mts.dtypes

#Checking Stationarity
#Johansen test can be used to check stationarity between maximum of 12 variables.
reuslt=coint_johansen(mts,0,1).lr1# trace value lr2 for e=max eigen values
reuslt
result2=coint_johansen(mts,0,1).cvt# cvm for critical eigen values
result2
#Critical trace values for 90%, 95% and 99% Interval
#det_order=0 means there is no time trend and if trace value is > critical value than rej H0 and there is coint
help(coint_johansen)

def cointegration_test(df, alpha=0.05): 
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df,0,1)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

cointegration_test(mts)

r=coint_johansen(mts,-1,1).eig
r

#Splitting the data 
train=mts[:int(0.8*(len(mts)))]
valid=mts[int(0.8*(len(mts))):]

#fitting the model
model=VAR(endog=train)
model_fit=model.fit()

#Make prediction on validation
prediction=model_fit.forecast(model_fit.y,steps=len(valid))
prediction

#converting predictions to dataframe
cols=mts.columns
pred = pd.DataFrame(index=range(0,len(prediction)),columns=[cols])
for j in range(0,6):
    for i in range(0, len(prediction)):
       pred.iloc[i][j] = prediction[i][j]

#check rmse
######Error
for i in cols:
    print('rmse value for', i, 'is : ',math.sqrt(mean_squared_error(pred[i], valid[i])))

#make final predictions
model = VAR(endog=mts)
model_fit = model.fit()
yhat = model_fit.forecast(model_fit.y, steps=1)
print(yhat)