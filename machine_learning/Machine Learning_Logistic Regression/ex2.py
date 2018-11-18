import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

def featureNormalize(X):
    mu = X.mean(0)
    sigma = X.std(0)
    X = (X-mu)/sigma
    return X, mu, sigma

def plotData(X, y):
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    pos = X[y==1]
    neg = X[y==0]
    plt.plot(pos[:,0], pos[:,1], 'k+', markersize=10, label='Admitted',)
    plt.plot(neg[:,0], neg[:,1], 'ko', markersize=10, markerfacecolor='yellow', label='Not admitted')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend()
    plt.show()

def sigmoid(x):
    return 1./(1+np.e**(-x))

def computeCost(X, y ,theta):
    J = 0
    m = y.shape[0]
    for i in range(m):
        g = sigmoid(np.dot(X[i,:], theta))
        J = J + y[i]*math.log(g)+(1-y[i])*math.log(1-g+1e-10)
    J = -1./m*J
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - alpha / m * np.dot(X.T, sigmoid(np.dot(X, theta)) - y)
        J_history[i] = computeCost(X, y, theta)
    return J_history, theta

def predict(X, theta, m):
    p = np.zeros((m,))
    for i in range(m):
        if sigmoid(np.dot(X[i], theta))>=0.5:
            p[i] = 1

    return p

data =  pd.read_csv('/root/Machine Learning/ex2/ex2data1.txt', header=None)

X = data.ix[:, 0:1]
y = data.ix[:, 2]
X = np.array(X)
y = np.array(y)
m = len(y)

#X, mu, sigma = featureNormalize(X)
print ('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples')
plotData(X, y)

alpha = 0.3
num_iters = 1000

X, mu, sigma = featureNormalize(X)
X = np.column_stack((np.ones((m,1)), X))
y = y.reshape((m,1))
theta = np.zeros((X.shape[1], 1))


J_history, theta = gradientDescent(X, y, theta, alpha, num_iters)

print (J_history[num_iters-1])
print (theta)
plt.plot(range(num_iters), J_history)
plt.show()

A = np.array([[45, 85]])
A = (A-mu)/sigma
A = np.column_stack(([1], A))
print (sigmoid(np.dot(A, theta)))


X = data.ix[:, 0:1]
y = data.ix[:, 2]
X = np.array(X)
y = np.array(y)
X, mu, sigma = featureNormalize(X)

m = len(y)
fig = plt.figure()
fig.add_subplot(1, 1, 1)
pos = X[y==1]
neg = X[y==0]
plt.plot(pos[:,0], pos[:,1], 'k+', markersize=10, label='Admitted',)
plt.plot(neg[:,0], neg[:,1], 'ko', markersize=10, markerfacecolor='yellow', label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
x = np.linspace(-2, 2)
plt.plot(x,(-theta[0]-theta[1]*x)/theta[2])
plt.show()

X = np.column_stack((np.ones((m, 1)),X))
p  = predict(X, theta, m)
print ('The accuracy is ', np.mean(p==y))