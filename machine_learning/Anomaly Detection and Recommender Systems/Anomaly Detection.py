import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.io as sio
from numpy import linalg
import math


def estimateGaussian(X):
    m, n = X.shape
    mu = np.zeros((n,))
    sigma2 = np.zeros((n,))
    for i in range(n):
        mu[i] = 1./m*sum(X[:,i])
        sigma2[i] =  1./m*sum((X[:,i]-mu[i])**2)

    return mu, sigma2

def multivariateGaussian(X, mu, Sigma2):
    m, n = X.shape
    Sigma2 = np.diag(Sigma2)
    p = np.zeros((m,))
    for i in range(m):
        p[i] = (2*math.pi)**(-n/2.)*(linalg.det(Sigma2)**-0.5)*np.e**np.dot((-0.5*np.dot((X[i,:].T-mu).T,linalg.inv(Sigma2))),X[i,:].T-mu)

    return p

def visualizeFit(X, mu, sigma2):
    x1 = np.linspace(0, 30,100)
    x2 = np.linspace(0, 30, 100)
    Z = np.zeros((100,100))
    for i in range(100):
        for j in range(100):
            x = np.zeros((1,2))
            x[0,0] = x1[i]
            x[0,1] = x2[j]
            Z[i,j] = multivariateGaussian(x, mu, sigma2)

    Z = Z.T
    plt.plot(X[:,0], X[:,1], 'bx')
    x1, x2 = np.meshgrid(x1, x2)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.contour(x1, x2, Z, [1e-20, 1e-17, 1e-14, 1e-11, 1e-8, 1e-5, 1e-2], color='blue')
    plt.show()

def selectThreshold(y, p):
    bestEpsilon = 0
    bestF1 = 0
    stepsize = (max(p)-min(p))/1000
    for epsilon in np.arange(min(p)+stepsize, max(p), stepsize):
        predictions = (p<epsilon)
        tp = sum((predictions==1) & (y==1))
        fp = sum((predictions==1) & (y==0))
        fn = sum((predictions==0) & (y==1))
        prec = float(tp)/(tp+fp)
        rec =  float(tp)/(tp+fn)
        F1 = (2*prec*rec)/(prec+rec)
        if (F1>bestF1):
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1

data1 = sio.loadmat('/root/Machine Learning/ex8/ex8data1.mat')

X = data1['X']
Xval = data1['Xval']
yval = data1['yval']
yval = yval.reshape((yval.shape[0],))
plt.plot(X[:,0], X[:,1], 'bx')
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

mu, sigma2 = estimateGaussian(X)

p = multivariateGaussian(X, mu, sigma2)
visualizeFit(X, mu ,sigma2)

pval = multivariateGaussian(Xval , mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)

print ('Best epsilon found using cross-validation: %e'%epsilon)
print ('Best F1 on Cross Validation Set: %f' %F1)

outliers = X[(p<epsilon)]

plt.plot(X[:,0], X[:,1], 'bx')
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.scatter(outliers[:,0], outliers[:,1], marker='o', linewidths=15, s=18, edgecolors='r')
plt.show()

data2 = sio.loadmat('/root/Machine Learning/ex8/ex8data2.mat')

X = data2['X']
Xval = data2['Xval']
yval = data2['yval']
yval = yval.reshape((yval.shape[0],))

mu, sigma2 = estimateGaussian(X)

p = multivariateGaussian(X, mu, sigma2)
pval = multivariateGaussian(Xval, mu, sigma2)

epsilon, F1 = selectThreshold(yval , pval)
print ('Best epsilon found using cross-validation: %e'%epsilon)
print ('Best F1 on Cross Validation Set: %f' %F1)
print ('# Outliers found: %d' %sum(p<epsilon))