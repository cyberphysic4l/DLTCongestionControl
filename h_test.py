import math
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt


h0 = 0.13
h1 = 1.13
h = h0+(h1-h0)/2
beta = math.sqrt(h1-h0)
ls = np.arange(0, 2, 0.001)
y = np.zeros(len(ls))
for i,l in enumerate(ls):
    integ = integrate.quad(lambda x: math.exp(-(x**2)/l), 0, beta)
    y[i] = h0 + (l/2)*math.exp(-(beta**2)/l) +beta*integ[0]

plt.plot(ls, y)
plt.plot(ls,ls)
plt.show()

'''
eps = 0.000001
dx = 0.001
alpha = 10
err = 1
while err>eps:
    integ = integrate.quad(lambda x: math.exp(-(x**2)/l), 0, beta)
    y = h0 + (l/2)*math.exp(-(beta**2)/l) +beta*integ[0]
    l1 = l+dx
    integ = integrate.quad(lambda x: math.exp(-(x**2)/l1), 0, beta)
    y1 = y0 = h0 + (l1/2)*math.exp(-(beta**2)/l1) +beta*integ[0]

    grad = (y1-y)/dx
    err = (y-l)**2
    l = l+alpha*grad*err
    print('l = ' + str(l))
    print('y = ' + str(y))
    print()'''




print('l = ' + str(y))

print('l = ' + str(y/h) +'h')

print('L = ' + str(l*250))