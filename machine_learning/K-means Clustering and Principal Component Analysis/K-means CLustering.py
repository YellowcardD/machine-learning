import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math
import scipy.io as sio

def findCloestCentorids(X, centroids):
    idx = np.zeros((m,))
    for i in range(m):
        min = 999999
        for k in range(1,K+1):
            if (sum((X[i,:]-centroids[k-1,:])**2)<min):
                min = sum((X[i,:]-centroids[k-1,:])**2)
                idx[i] = k

    return idx

def computeCentroids(X, idx, K):
    centroids = np.zeros((K, n))
    for k in range(1, K+1):
        cnt = 0
        all = np.zeros((n,))
        for i in range(m):
            if(idx[i]==k):
                cnt = cnt + 1
                all = all + X[i,:]

        centroids[k-1] = all/cnt

    return centroids

def plotProgresskMeans(X, centroids, idx, K, i):
    plt.subplot(2,3,i)
    plt.title('iteration %d' %i)
    plt.scatter(centroids[:,0], centroids[:,1],marker='X', color='k', s=50)
    for i in range(m):
        if idx[i]== 1:
            plt.plot(X[i,0], X[i,1], Marker='o', color='r')
        elif idx[i]==2:
            plt.plot(X[i,0], X[i,1], Marker='o', color='y')
        else:
            plt.plot(X[i,0], X[i,1], Marker='o', color='b')

def runkMeans(X, initial_centroids, max_iters):
    centroids = initial_centroids
    for i in range(1,max_iters+1):
        print ('K-Means iteration %d/%d' %(i, max_iters))
        idx = findCloestCentorids(X, centroids)
        plotProgresskMeans(X, centroids, idx, K, i)
        centroids = computeCentroids(X, idx, K)

    plt.show()
    return centroids,idx

data = sio.loadmat('/root/Machine Learning/ex7/ex7data2.mat')
X  = data['X']
m = X.shape[0]
n = X.shape[1]


K = 3
initial_centroids = np.array([[3, 3],[6, 2],[8, 5]])

idx = findCloestCentorids(X, initial_centroids)
centroids = computeCentroids(X, idx, K)

max_iters = 6
centroids, idx = runkMeans(X, initial_centroids, max_iters)

