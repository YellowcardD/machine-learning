import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import *

def featureNormalize(X):
    X_norm = X
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    mu = X.mean(0)
    sigma = X.std(0)
    X = (X-mu)/sigma
    return X, mu, sigma

def computeCostMulti(X, y, theta):
    return 1./ 2 /m * sum(np.dot((np.dot(X, theta) - y).T, np.dot(X, theta) - y))

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - alpha / m * np.dot(X.T, np.dot(X, theta) - y)
        J_history[iter] = computeCostMulti(X, y, theta)

    return theta, J_history

def normalEqn(X, y):
    return np.dot(np.dot(mat(np.dot(X.T, X)).I, X.T), y)

data = pd.read_csv('/root/Machine Learning/ex1/ex1data2.txt', header=None)

X = data.ix[:, 0:1]
X = np.array(X)
y = data.ix[:, 2]
m = len(y)
y = y.reshape((m,1))

#print 'First 10 examples from the dataset: '
#for i in range(10):
#    print ('x = [%.0f, %.0f], y = %.0f' %(X[i, 0], X[i, 1], y[i, :]))

X, mu, sigma = featureNormalize(X)
X = np.column_stack((np.ones((m,1)), X))

#theta = np.zeros((X.shape[1], 1))

alpha = [0.3, 0.1, 0.03, 0.01, 0.003]
colors = ['black', 'blue', 'green', 'yellow', 'red']
num_iters = 50

fig = plt.figure()
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.xlim([0, 50])
for i in range(len(alpha)):
    theta = np.zeros((X.shape[1], 1))
    theta, J_history = gradientDescentMulti(X, y, theta, alpha[i], num_iters)
    plt.plot(np.arange(num_iters).reshape((num_iters, 1)), J_history, color=colors[i], label='alpha = %.3f' %alpha[i])

plt.legend()
plt.show()

print ('Theta computed from gradient descent:')
print (theta)
A = np.array([[1650 ,3]])
A = (A-mu)/sigma
A = np.column_stack(([1], A))
price = np.dot(A, theta)
print ('Predict price of a 1650 sq-ft, 3 br house (using gradient descent) $%f' %price)


X = data.ix[:, 0:1]
X = np.array(X)
X = np.column_stack((np.ones((m,1)), X))
theta = normalEqn(X, y)

print ('Theta computed from the normal equations')
print (theta)

B=np.array([[1, 1650, 3]])
price = np.dot(B, theta)
print ('Predicted price of a 1650 sq-ft, 3 br house (using normal equations $%f' %price)