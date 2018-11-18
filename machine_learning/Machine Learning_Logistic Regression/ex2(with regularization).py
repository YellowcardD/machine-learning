import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

def plotData(X, y):
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    pos = X[y==1]
    neg = X[y==0]
    plt.plot(pos[:,0], pos[:,1], 'k+', markersize=7, label='y=1')
    plt.plot(neg[:,0], neg[:,1], 'ko', markerfacecolor='yellow', markersize=7, label='y=0')
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.show()

def featureNormalize(X):
    mu = X.mean(0)
    sigma = X.std(0)
    X = (X-mu)/sigma
    return X, mu, sigma

def mapFeature(X1, X2):
    degree = 6
    X1 = X1.reshape((len(X1),1))
    X2 = X2.reshape((len(X2),1))
    out = np.ones((len(X1),1))
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.column_stack((out, (X1**(i-j))*(X2**(j))))

    return out

def sigmoid(x):
    return 1. / (1 + np.e ** (-x))

def costFunctionReg(X, y, theta, Lambda):
    J = 0;
    J1 = 0
    J2 = 0
    m = len(y)
    for i in range(m):
        g = sigmoid(np.dot(X[i,:], theta))
        J1 = J1 + y[i]*math.log(g) + (1-y[i])*math.log(1-g+1e-10)
    J1 = -1./m*J1
    for i in range(1, theta.shape[0]):
        J2 = J2 + theta[i]**2
    J2 = Lambda/2./m*J2
    J = J1 + J2
    return  J

def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - alpha / m * (np.dot(X.T, sigmoid(np.dot(X, theta)) - y)+ Lambda*theta)
        theta[0] = theta[0] + alpha*Lambda/m*theta[0]
        J_history[i] = costFunctionReg(X, y, theta, Lambda)
    return theta, J_history

def predict(X, theta, m):
    p = np.zeros((m,1))
    for i in range(m):
        if sigmoid(np.dot(X[i], theta)) >= 0.5:
            p[i] = 1

    return p

data = pd.read_csv('/root/Machine Learning/ex2/ex2data2.txt', header=None)

X = data.ix[:,0:1]
y = data.ix[:,2]
X = np.array(X)
y = np.array(y)
m = len(y)
#print X.shape,y.shape

plotData(X, y)

X = mapFeature(X[:,0], X[:,1])
X, mu, sigma = featureNormalize(X)

X[:,0] = np.ones((m,))
y = y.reshape((m,1))
#print X.shape
initial_theta = np.zeros((X.shape[1],1))
Lambda = 1.0
print (costFunctionReg(X, y, initial_theta, Lambda))

alpha = 0.3
num_iters = 1000

theta, J_history = gradientDescent(X, y, initial_theta, alpha, num_iters)
print (J_history[num_iters-1])
#print theta.shape
plt.plot(range(num_iters), J_history)
plt.show()



p = predict(X, theta, m)
print ('The accuracy is ', np.mean(p==y))