import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.io as sio


def polyFeatures(X, p):

    for i in range(2, p+1):
        X = np.column_stack((X, X[:,0]**i))

    return X

def learningCurve(X, y, Xval, yval, alpha, theta, Lambda):
    error_train = np.zeros((m+1,))
    error_val = np.zeros((m+1,))
    num_iters = 10000
    for i in range(2,m+1):
        Y = y[0:i]
        theta, J_history= gradientDescent(X[0:i,:], Y, theta, alpha, num_iters, Lambda)
        error_train[i] = linearRegCostFunction(X[0:i,:], Y, theta, 0)
        error_val[i] = linearRegCostFunction(Xval, yval, theta, 0)

    return error_train, error_val

def gradientDescent(X, y, theta, alpha, num_iters, Lambda):
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - alpha *(1./m * np.dot(X.T, np.dot(X, theta) - y) + Lambda/m*theta)
        theta[0] =  theta[0]+alpha*Lambda/m*theta[0]
        #J_history[iter] = linearRegCostFunction(X, y, theta, Lambda)

    return theta, J_history

def linearRegCostFunction(X, y, theta, Lambda):
    m = X.shape[0]
    J = 1./2/m*np.dot((np.dot(X, theta)-y).T, (np.dot(X, theta)-y))
    J = J + 1./2/m*Lambda*sum(theta[1:]**2)
    return J

def featureNormalize(X):
    mu = X.mean(0)
    sigma = X.std(0)
    X = (X-mu)/sigma
    return X, mu, sigma

def validationCurve(X, y, Xval, yval, theta, alpha, num_iters):
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    error_train = np.zeros((len(lambda_vec),))
    error_val = np.zeros((len(lambda_vec),))
    for i in range(len(lambda_vec)):
        theta, J_history= gradientDescent(X, y, theta, alpha, num_iters, lambda_vec[i])
        error_train[i] = linearRegCostFunction(X, y, theta, 0)
        error_val[i] = linearRegCostFunction(Xval, yval, theta, 0)

    return lambda_vec, error_train, error_val

data = sio.loadmat('/root/Machine Learning/ex5/ex5data1.mat')

X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']
m = X.shape[0]
n = X.shape[1]

p = 8
X_poly = polyFeatures(X, p)
X_poly, mu, sigma= featureNormalize(X_poly)
X_poly = np.column_stack((np.ones((X_poly.shape[0], 1)),X_poly))
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = (X_poly_test-mu)/sigma
X_poly_test = np.column_stack((np.ones((X_poly_test.shape[0],1)),X_poly_test))
X_poly_val = polyFeatures(Xval, p)
X_poly_val = (X_poly_val-mu)/sigma
X_poly_val = np.column_stack((np.ones((X_poly_val.shape[0],1)),X_poly_val))
'''
Lambda = 0
alpha = 0.01
num_iters = 10000
initial_theta = np.ones((p+1,1))
theta, J_history= gradientDescent(X_poly, y, initial_theta, alpha, num_iters, Lambda)

plt.plot(X, y, 'rx', 10)
x = np.arange(min(X)-15, max(X)+25, 0.05)
x_map = x.reshape((len(x),1))
x_map = polyFeatures(x_map, p)
x_map = (x_map-mu)/sigma
x_map = np.column_stack((np.ones((len(x),1)),x_map))
plt.plot(x, np.dot(x_map,theta), 'b--')
plt.ylim([-60, 40])
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title(('Polynomial Regression Fit (lambda = %f)' %Lambda))
plt.show()

initial_theta = np.ones((p+1,1))
error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, alpha, initial_theta, Lambda)
plt.plot(np.arange(2, m+1), error_train[2:m+1], label='Train')
plt.plot(np.arange(2, m+1), error_val[2:m+1], label='Cross Validation')
plt.title('Polynomial Regression Learning Curve (lambda = %f)' %Lambda)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim([0, 13])
plt.ylim([0, 100])
plt.legend()
plt.show()

Lambda = [1, 100]
for i in range(2):

    theta, J_history= gradientDescent(X_poly, y, initial_theta, 0.001, num_iters, Lambda[i])
    plt.plot(X, y, 'rx', 10)
    x = np.arange(min(X)-15, max(X)+25, 0.05)
    x_map = x.reshape((len(x),1))
    x_map = polyFeatures(x_map, p)
    x_map = (x_map-mu)/sigma
    x_map = np.column_stack((np.ones((len(x),1)),x_map))
    plt.plot(x, np.dot(x_map,theta), 'b--')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title(('Polynomial Regression Fit (lambda = %f)' %Lambda[i]))
    plt.show()

    error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, 0.001, initial_theta, Lambda[i])
    plt.plot(np.arange(2, m + 1), error_train[2:m + 1], label='Train')
    plt.plot(np.arange(2, m + 1), error_val[2:m + 1], label='Cross Validation')
    plt.title('Polynomial Regression Learning Curve (lambda = %f)' % Lambda[i])
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    #plt.xlim([0, 13])
    #plt.ylim([0, 100])
    plt.legend()
    plt.show()

'''
initial_theta = np.ones((p+1,1))
num_iters = 30000
lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval, initial_theta, 0.00001, num_iters)
plt.plot(lambda_vec, error_train, label='Train')
plt.plot(lambda_vec, error_val, label='Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.ylim([0, 20])
plt.legend()
plt.show()

theta, J_history= gradientDescent(X_poly, y, initial_theta, 0.0001, num_iters, 3)
print (linearRegCostFunction(X_poly_test, ytest, theta, 0))