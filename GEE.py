import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import math
import pandas as pd
import random

# data = sm.datasets.get_rdataset("epil", "MASS").data
# print(data)
# md = smf.gee("y ~ age + trt + base", "subject", data,
#              cov_struct=sm.cov_struct.Independence(),
#              family=sm.families.Poisson())
# mdf = md.fit()
# print(mdf.summary())
y=[]
x2=[[xx] for xx in np.arange(0.,100.,1.)]
for kk in x2:
    y.append(0.5*kk[0]+3+10*math.sin(kk[0]/3.14)*random.random())
panddf=pd.DataFrame({'x2':x2,'y':y})

md2 = smf.gee("x2", "y", panddf,
             cov_struct=sm.cov_struct.Independence(),
             family=sm.families.Poisson())
mdf2 = md2.fit()
print(mdf2.summary())