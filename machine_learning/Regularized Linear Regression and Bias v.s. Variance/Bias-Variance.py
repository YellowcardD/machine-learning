import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.io as sio

def linearRegCostFunction(X, y, theta, Lambda):
    m = X.shape[0]
    J = 1./2/m*np.dot((np.dot(X, theta)-y).T, (np.dot(X, theta)-y))
    J = J + 1./2/m*Lambda*sum(theta[1:]**2)
    return J

def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - alpha *(1./m * np.dot(X.T, np.dot(X, theta) - y) + Lambda/m*theta)
        theta[0] =  theta[0]+alpha*Lambda/m*theta[0]
        #J_history[iter] = linearRegCostFunction(X, y, theta, Lambda)

    return theta, J_history

def learningCurve(X, y, Xval, yval, theta, Lambda):
    error_train = np.zeros((m+1,))
    error_val = np.zeros((m+1,))
    alpha = 0.001
    num_iters = 5000
    for i in range(2,m+1):
        Y = y[0:i]
        theta, J_history= gradientDescent(X[0:i,:], Y, theta, alpha, num_iters, Lambda)
        error_train[i] = linearRegCostFunction(X[0:i,:], Y, theta, 0)
        error_val[i] = linearRegCostFunction(Xval, yval, theta, 0)

    return error_train, error_val

def polyFeatures(X, p):

    for i in range(2, p+1):
        X = np.column_stack((X, X[:,0]**i))

    return X

def featureNormalize(X):
    mu = X.mean(0)
    sigma = X.std(0)
    X = (X-mu)/sigma
    return X, mu, sigma

data = sio.loadmat('/root/Machine Learning/ex5/ex5data1.mat')

X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
m = X.shape[0]
n = X.shape[1]

Lambda = 1
initial_theta = np.ones((n+1,1))
error_train, error_val = learningCurve(np.column_stack((np.ones((m, 1)), X)), y, np.column_stack((np.ones((Xval.shape[0], 1)), Xval)), yval, initial_theta, Lambda)
plt.plot(range(2,m+1), error_train[2:m+1], label='Train')
plt.plot(range(2,m+1), error_val[2:m+1], label='Cross Validation')
plt.title('Learning curve for linear regression')
plt.legend()
plt.xlabel('Number of training examples')
plt.ylabel('error')
plt.xlim([0, 13])
plt.ylim([0, 150])
plt.show()

#for i in range(2,m+1):
#   print error_train[i], error_val[i]

