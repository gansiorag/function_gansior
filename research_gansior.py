import base_function
import numpy as np
import random
import matplotlib.pyplot as plt
def matrparam(vid):
    if vid==0: matr=np.array([[3., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.]])

    if vid==1: matr=np.array([[3., 1., 0.00, 0., 0., 0.],
                              [3., 1., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 1., 0., 0., 0., 0.]])

    if vid==2: matr=np.array([[3., 1., 0., 0., 0., 0.],
                              [3., 1., 0.0, 0., 0., 0.],
                              [4., 1., 5., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.]])
    return matr


if __name__=="__main__":
    #prim=np.random.randint(10.,60.,111)
    #print(prim)
    y0=[]
    y1=[]
    step=1
    x=range(0,50,1)
    for i in x:
        y0.append(base_function.gansior(matrparam(1), "gauss", i, step))
        y1.append(base_function.gansior(matrparam(2), "gauss", i, step))
    plt.plot(x, y0, 'r^',markersize=10)
    plt.plot(x, y0, 'y--', linewidth=1)
    plt.plot(x,y1,'bo--', linewidth=1)
    plt.show()


