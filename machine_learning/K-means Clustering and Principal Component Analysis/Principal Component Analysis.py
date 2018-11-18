import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.io as sio
from numpy import linalg

def featureNormalize(X):
    mu = X.mean(axis=0)
    X = X-mu
    sigma = X.std(axis=0)
    X = X/sigma
    return X, mu ,sigma

def pca(X):
    m = X.shape[0]
    Sigma = np.dot(X.T, X)/m
    return linalg.svd(Sigma)

def projectData(X, U , K):
    m = X.shape[0]
    Z = np.zeros((m,K))
    for i in range(m):
        for k in range(K):
            Z[i,k] = np.dot(X[i,:], U[:,k])

    return Z

def recoverData(Z, U, K):
    U_reduce = U[:,0:K]
    X_rec = np.dot(Z, U_reduce.T)
    return X_rec

def displayData(X):
    m = X.shape[0]
    for i in range(1,m+1):
        plt.subplot(10, 10, i)
        plt.imshow(X[i-1,:].reshape((32,32)).T,cmap = 'gray')
        plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

'''
data = sio.loadmat('/root/Machine Learning/ex7/ex7data1.mat')
X = data['X']
m = X.shape[0]
n = X.shape[1]
plt.plot(X[:,0], X[:,1], 'bo')
plt.axis([0.5, 6.5, 2, 8])
plt.show()

X_norm, mu, sigma= featureNormalize(X)

U, S, V = pca(X_norm)

plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
plt.show()

K = 1
Z = projectData(X_norm, U, K)

X_rec = recoverData(Z, U, K)

plt.plot(X_norm[:,0], X_norm[:,1], 'bo')
plt.plot(X_rec[:,0], X_rec[:,1], 'ro')
plt.axis('square')
plt.axis([-4, 3, -4, 3])
for i in range(m):
    plt.plot([X_norm[i,0],X_rec[i,0]], [X_norm[i,1],X_rec[i,1]], '--k',linewidth=1)

plt.show()
'''
data1 = sio.loadmat('/root/Machine Learning/ex7/ex7faces.mat')
X = data1['X']

displayData(X[0:100,:])

X_norm, mu, sigma = featureNormalize(X)

U, S, V = pca(X_norm)
displayData(U[:,0:36].T)


K = 100
Z = projectData(X_norm, U, K)
X_rec = recoverData(Z, U, K)
plt.subplot(1, 2, 1)
displayData(X_norm[0:100,:])
plt.subplot(1, 2, 2)
displayData(X_rec[0:100,:])
