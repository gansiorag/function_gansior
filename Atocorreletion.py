import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sklearn.linear_model as ln
from sklearn.model_selection import KFold
import sklearn.model_selection as model
from matplotlib.cbook import get_sample_data
import warnings
import itertools
import statsmodels.api as sm
import pandas as pd
import datetime

plt.style.use('fivethirtyeight')

fs=ln.LinearRegression()


x2=[[xx] for xx in np.arange(0.,100.,1.)]
y=[]
for kk in x2:
    y.append(0.4*kk[0]+3+6*math.sin(kk[0]/3.14))
print(y)
ydata=pd.Series(y,name='Model')
print(ydata)
a=datetime.datetime.today()
dateList = []
# for dd in range(0,100):
#     dateList.append(a-datetime.timedelta(days=dd))
start = datetime.datetime(2019,3,1)
dategener= [start + datetime.timedelta(days= dd) for dd in range(0,100)]
for dd in dategener:
    dateList.append(dd.strftime("%Y-%m-%d"))
dateIsh=pd.DataFrame({'Date':dateList,
                      'z':y},columns=['Date','z'],index=None)
dateIsh.reset_index(inplace=True)
dateIsh['Date'] = pd.to_datetime(dateIsh['Date'])
dateIsh = dateIsh.set_index('Date')
#print(dateIsh)
print(dateIsh.describe())
#print(dateIsh.isnull().sum())
from pylab import rcParams
rcParams['figure.figsize']=11,9
#dateIsh.plot(figsize=(15, 6))
#dfd=dateIsh.resample('D',on='Date').mean()
#print(dfd)
decomposition=sm.tsa.seasonal_decompose(dateIsh.z,model='additive')
print("Trend = ",decomposition.seasonal )
#plt.plot(decomposition.seasonal)
fig = decomposition.plot()
plt.show()
#x2=np.column_stack((x,y))
#print(x2)
#print(y)

# x1=np.random.choice(np.linspace(40,45,100),50)
# y1=0.4*x1+3+6*np.sin(x1/3.14)


#x_train, x_test, y_train, y_test =model._split(x2,y,0.33)
fs.fit(x2,y)
print(fs.intercept_)
print(fs.coef_)
# b0, b1=fs.coef_
# print(b0,b1)
print(fs.score(x2,y))
# p = np.polyfit(x2[:,0],y,1)
# f=np.poly1d(p)
# y2=f(x2[:,0])
#plt.plot(x2[:,0],fs.predict(x2), c='b')
y2=fs.predict(x2)
#plt.scatter(x2,y2,c='g')
choice_num=[]
num_max=23
a=1.
c=17.
x0=3.
choice_num.append(x0)
for i in range(1,num_max):
    choice_num.append((choice_num[i-1]*a+c) % num_max)

print(choice_num)
#plt.axhline(20,0,100,linestyle=':',c='r')
#plt.plot(x2,y,'yo',x2,y2,'--k')
#plt.scatter(x2,y, c='b')
#plt.scatter(x1,y1, c='r')
# Определите p, d и q в диапазоне 0-2
p = d = q = range(0, 2)
# Сгенерируйте различные комбинации p, q и q
pdq = list(itertools.product(p, d, q))
# Сгенерируйте комбинации сезонных параметров p, q и q
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
warnings.filterwarnings("ignore") # отключает предупреждения
'''
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                order=param,
                seasonal_order=param_seasonal,
                enforce_stationarity=False,
                enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
            '''
mod1 = sm.tsa.statespace.SARIMAX(y,
    order=(0, 0, 0),
    seasonal_order=(1, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False)
results2 = mod1.fit()
print(results2.summary().tables[1])
results2.plot_diagnostics(figsize=(15,12))

#plt.show()
