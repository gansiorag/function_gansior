import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sklearn.linear_model as ln
from sklearn.model_selection import KFold
import sklearn.model_selection as model
from matplotlib.cbook import get_sample_data

fs=ln.LinearRegression()


x2=[[xx] for xx in np.arange(0.,100.,1.)]
y=[]
for kk in x2:
    y.append(0.4*kk[0]+3+6*math.sin(kk[0]/3.14))
print(y)
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
#plt.show()
