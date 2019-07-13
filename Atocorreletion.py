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
import countStat as CS

plt.style.use('fivethirtyeight')

fs=ln.LinearRegression()

"""
https://datahub.io/core/finance-vix/r/vix-daily.json

Field information
Field Name	Order	Type (Format)	Description
Date	1	date (%Y-%m-%d)	
VIX Open	2	number (default)	
VIX High	3	number (default)	
VIX Low	4	number (default)	
VIX Close	5	number (default)

"""


fig2=plt.figure()
ax2=fig2.add_subplot(211)

#fig3=plt.figure()
ax3=fig2.add_subplot(311)

#ax1=fig1.add_subplot(111)


#ax2=fig1.add_subplot(211)


#ax3=fig1.add_subplot(311)


x2=[[xx] for xx in np.arange(0.,100.,1.)]
x3=x2.copy()
#random.shuffle(x2)

#y=[6.0,4.4,5.0,9.0,7.2,4.8,6.0,10.0,8.0,5.6,6.4,11.0,9.0,6.6,7.0,10.8]
#print(y)

y=[]
for kk in x3:
    y.append(0.5*kk[0]+3+10*math.sin(kk[0]/3.14))

fig1=plt.figure(1)
#plt.plot(x3, y,c='y')
plt.title("Periodic curve with trend")
plt.scatter(x3, y, c='r')
#XX=np.column_stack((x3))
XX = np.array(x3).reshape((-1,1))
fs.fit(XX,y)
yy=fs.predict(XX)
print('coef b0 :', fs.intercept_)
print('coef b1 :', fs.coef_)
print('determin', fs.score(XX,y))

plt.plot(x3,yy,'g-')

fig11=plt.figure(11)
coef=CS.auto_corr(y)
#print("fig0 ", coef)
rr=coef.pop(0)
rr=coef.pop(0)
print(max(coef))
x7=range(1,len(coef)+1)
plt.plot(x7,coef)
plt.scatter(x7,coef, c='r')

y1=[]
for kk in x3:
    y1.append(0.5*kk[0]+3+10*math.sin(kk[0]/3.14)*random.random())
fig2=plt.figure(2)
plt.plot(x3, y1)
plt.title("Periodic curve with trend and random amplitude changes")
fs.fit(XX,y1)
yy1=fs.predict(XX)
plt.plot(x3, yy1)
print('coef b0 :', fs.intercept_)
print('coef b1 :', fs.coef_)
print('determin', fs.score(XX,y1))

fig12=plt.figure(12)
coef=CS.auto_corr(y)
#print("fig1 ", coef)
rr=coef.pop(0)
rr=coef.pop(0)
print(max(coef))
x8=range(1,len(coef)+1)
plt.plot(x8,coef)
plt.scatter(x8,coef, c='r')

y2=[]
for kk in x3:
    y2.append(0.5*kk[0]+3+10*math.sin(kk[0]*random.random()/3.14))
fig3=plt.figure(3)
plt.plot(x3, y2)
plt.title("Periodic curve with trend and random periodic changes")

fig13=plt.figure(13)
coef=CS.auto_corr(y)
#print("fig3 ", coef)
rr=coef.pop(0)
rr=coef.pop(0)
print(max(coef))
x9=range(1,len(coef)+1)
plt.plot(x9,coef)
plt.scatter(x9,coef, c='r')



y3=[]
for kk in x3:
    y3.append(0.5*kk[0]+3*random.random()+10*math.sin(kk[0]/3.14))
fig4=plt.figure(4)
plt.plot(x3, y3)
plt.title("Periodic curve with trend and random free path changes")

fig14=plt.figure(14)
coef=CS.auto_corr(y)
#print("fig4 ", coef)
rr=coef.pop(0)
rr=coef.pop(0)
print(max(coef))
x10=range(1,len(coef)+1)
plt.plot(x10,coef)
plt.scatter(x10,coef, c='r')


# for kk in x3:
#     y3.append(0.5*kk[0]+3+10*math.sin(kk[0]/3.14)*random.random())
#y=[6.0,4.4,5.0,9.0,7.2,4.8,6.0,10.0,8.0,5.6,6.4,11.0,9.0,6.6,7.0,10.8]

#print(y)
ax2.scatter(x2, y2)
#ax3.scatter(x3, y3, c='r')


y4=[]
for kk in x3:
    y4.append(0.5*kk[0]*random.random()+3+10*math.sin(kk[0]/3.14))
fig5=plt.figure(5)
plt.plot(x3, y4)
plt.title("Periodic curve with trend and random corner changes")

fig10=plt.figure(10)
coef=CS.auto_corr(y4)
#print("fig5 ", coef)
rr=coef.pop(0)
rr=coef.pop(0)
print(max(coef))
x7=range(1,len(coef)+1)
plt.plot(x7,coef)



y5=[]
for kk in x3:
    y5.append(0.5*kk[0]*random.random()+3*random.random()+10*math.sin(kk[0]/3.14))
fig6=plt.figure(6)
plt.plot(x3, y5)
plt.title("Periodic curve with trend and random corner changes")

fig9=plt.figure(9)
coef=CS.auto_corr(y5)
#print("fig6 ", coef)
rr=coef.pop(0)
rr=coef.pop(0)
print(max(coef))
x6=range(1,len(coef)+1)
plt.plot(x6,coef)



ax1.plot(x,coef)
panddf=pd.DataFrame({'x2':x2,'y':y})
print(panddf)
decomposition=sm.tsa.seasonal_decompose(panddf['y'],model='additive')
print("Trend = ",decomposition.seasonal )
ax2.plot(decomposition.seasonal)

y6=[]
for kk in x3:
    y6.append(0.7*kk[0]*random.random()+3*random.random()+
              10*math.sin(kk[0]*random.random()/3.14)*random.random())
fig7=plt.figure(7)
plt.plot(x3, y6)
plt.title("Periodic curve with trend and random corner changes")

fig8=plt.figure(8)
coef=CS.auto_corr(y6)
#print("fig7 ", coef)
rr=coef.pop(0)
rr=coef.pop(0)
print(max(coef))
x=range(1,len(coef)+1)
plt.plot(x,coef)



#plt.plot(x,coef)

"""
#------------ end 1. ----------

ydata=pd.Series(y,name='Model') # преобоазуем в массив pfndas.Series
#print(ydata)
dateList = []

start = datetime.datetime(2019,3,1)
dategener= [start + datetime.timedelta(days= dd) for dd in range(0,100)]

for dd in dategener:
    dateList.append(dd.strftime("%Y-%m-%d"))

dateIsh=pd.DataFrame({'Date':dateList,'z':y}, columns=['Date','z'], index=None)
dateIsh.reset_index(inplace=None)
dateIsh['Date'] = pd.to_datetime(dateIsh['Date'])
dateIsh = dateIsh.set_index('Date')
print(dateIsh)
print(dateIsh['z'].describe())
print("sum - ", dateIsh['z'].sum())

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
"""
plt.show()