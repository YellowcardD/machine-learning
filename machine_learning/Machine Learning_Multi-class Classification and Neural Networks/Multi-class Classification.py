import scipy.io as sio
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

def displayData(X):
    fig, axes = plt.subplots(10, 10)
    k = 0
    for i in range(10):
        for j in range(10):
            axes[i,j].imshow(X[k,:].reshape((20,20)).T, plt.cm.gray)
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            k = k+1

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def sigmoid(x):
    return 1. / (1 + np.e ** (-x))

def costFunctionReg(X, y, theta, Lambda):
    J = 0
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

def gradientDescent(X, y, theta, alpha, num_iters, Lambda):

    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        theta = theta - alpha / m * (np.dot(X.T, sigmoid(np.dot(X, theta)) - y)+ Lambda*theta)
        theta[0] = theta[0] + alpha*Lambda/m*theta[0]
        J_history[i] = costFunctionReg(X, y, theta, Lambda)
    return theta.T, J_history

def oneVsAll(X, y, num_labels, Lambda):
    m = X.shape[0]
    n = X.shape[1]
    all_theta = np.zeros((num_labels, n+1))
    X = np.column_stack((np.ones((m,1)), X))
    initial_theta = np.zeros((n+1, 1))
    num_iters = 100
    alpha = 0.1
    for i in range(num_labels):
        all_theta[i,:], J_history= gradientDescent(X, y==i, initial_theta, alpha, num_iters, Lambda)

    return  all_theta

def predictOneVsAll(all_theta, X):

    X = np.column_stack((np.ones((X.shape[0], 1)), X))
    p = np.zeros((m,1))
    temp = sigmoid(np.dot(X, all_theta.T))
    p = np.argmax(temp,1)
    return p

data = sio.loadmat('/root/Machine Learning/ex3/ex3data1.mat')

X = data['X']
y = data['y']
for i in range(y.shape[0]):
    if y[i]==[10]:
        y[i]=[0]

input_layer_size = 400
num_labels = 10
m = X.shape[0]

rand_indices = range(m)
random.shuffle(rand_indices)

sel = X[rand_indices[0:100],:]
#displayData(sel)

Lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, Lambda)

pred = predictOneVsAll(all_theta, X)
print ('Training Set Accuracy:',np.mean(pred==y.reshape((m,))))