import numpy as np
import math

def auto_corr(y):
    array_coef=[]
    dl=len(y)

    col_koef= 1+math.ceil(dl/4)
    print(col_koef)
    y2 = []
    y1=list(y)
    for i in range(1,col_koef):
        y1.pop()
        y2 = list(y[i:])
        array_coef.append(math.fabs(np.corrcoef(y1,y2)[0,1]))


    return array_coef