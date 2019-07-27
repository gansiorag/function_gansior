import math

def degree(chh,step):
    dd=chh
    for _ in range(step-1):
        dd=dd*chh
    return dd

def bernuly(p,sob,isp):
    q=1-p
    ddd=degree(p,sob)*degree(q,isp-sob)*(math.factorial(isp)/(math.factorial(sob)*math.factorial(isp-sob)))
    return ddd

def laplas(p,sob,isp):
    q=1-p
    kf=1/math.sqrt(isp*p*q*2*math.pi)
    xf=(sob-isp*p)/math.sqrt(isp*p*q)
    xst=xf*xf/2
    ff=1/pow(math.e,xst)
    return kf*ff

def elem_gansior(list_koef, namefunc, x):
    """
    elementary step function gansior
    :param list_koef: list сoefficients from matrix function gansior
              matrix=np.array([[3., 1., 0., 0., 0., 0.],
                              [3., 1., 0.0, 0., 0., 0.],
                              [0., 1., 5., 0., 1., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.],
                              [0., 0., 0., 0., 0., 0.]])



    :param namefunc:  a function of a random variable "gauss" or other
    :param x: volue
    :return:
    """
    import random
    if namefunc=="gauss":
        result=list_koef[0] * random.gauss(list_koef[1],list_koef[2]) *\
               math.cos(list_koef[3] * random.gauss(list_koef[4],list_koef[5]) * x/math.pi)
    return result

def gansior(matkoef,matfunc,x,step):
    """
     function gansior in general form
     y=
    a00 * f00(a01,a02) * cos( a03 * f01(a04,a05) * x/pi) * F(x) +  - function trend
    a10 * f10(a11,a12) * cos( a13 * f11(a14,a15) * x/pi)     +     - error
    a20 * f20(a21,a22) * cos( a23 * f21(a24,a25) * x/pi)     +     - fluctuations in the minute
    a30 * f30(a31,a32) * cos( a33 * f31(a34,a35) * x/pi)     +     - fluctuations in the hour
    a40 * f40(a41,a42) * cos( a23 * f41(a44,a45) * x/pi)     +     - fluctuations in the day
    a50 * f50(a51,a52) * cos( a53 * f51(a54,a55) * x/pi)     +     - fluctuations in the week
    a60 * f61(a61,a62) * cos( a63 * f61(a64,a65) * x/pi)     +     - fluctuations in the mouns
    a70 * f70(a71,a72) * cos( a73 * f71(a74,a75) * x/pi)     +     - fluctuations in the four mouns
    a80 * f80(a81,a82) * cos( a83 * f81(a84,a85) * x/pi)           - fluctuations in the yaer

    :param matkoef: matrix coefficients
    :param matfunc: definition function "gauss"
    :param x: variable
    :param step: degree function trend
    :return:
    """
    result=elem_gansior(matkoef[0, :], matfunc, x)*pow(x,step)
    for i in range(1,matkoef.shape[0]):
        result+=elem_gansior(matkoef[i, :], matfunc, x)
    return result

if __name__=="__main__":
    p=1/3
    sob=10
    isp=30
    print("Bernully : ", bernuly(p, sob, isp))
    print("Degree : ", degree(2,3))
    print("pow : ", pow(2,3))
    print("Laplas : ", laplas(p,sob,isp))