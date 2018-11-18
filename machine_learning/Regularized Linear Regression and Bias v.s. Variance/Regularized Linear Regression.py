import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.io as sio

data = sio.loadmat('/root/Machine Learning/ex5/ex5data1.mat')

def PlotTrain():
    plt.plot(X, y, 'rx', 10)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()

def Plotfit():
    plt.plot(X, y, 'rx', 10)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    XX = np.column_stack((np.ones((m,1)), X))
    plt.plot(X, np.dot(XX, theta), '--')
    plt.show()

def linearRegCostFunction(X, y, theta, Lambda):
    J = 1./2/m*np.dot((np.dot(X, theta)-y).T, (np.dot(X, theta)-y))
    J = J + 1./2/m*Lambda*sum(theta[1:]**2)
    return J

def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - alpha *(1./m * np.dot(X.T, np.dot(X, theta) - y) + Lambda/m*theta)
        theta[0] =  theta[0]+alpha*Lambda/m*theta[0]
        J_history[iter] = linearRegCostFunction(X, y, theta, Lambda)

    return theta, J_history

X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
m = X.shape[0]
n = X.shape[1]
#PlotTrain()

theta = np.ones((n+1,1))
#J = linearRegCostFunction(np.column_stack((np.ones((m, 1)), X)), y, theta, 1)

Lambda = 0
alpha = 0.001
num_iters = 3000
theta, J_history= gradientDescent(np.column_stack((np.ones((m, 1)), X)), y, theta, alpha, num_iters, Lambda)
#print theta
#plt.plot(range(num_iters), J_history)
#plt.show()
Plotfit()