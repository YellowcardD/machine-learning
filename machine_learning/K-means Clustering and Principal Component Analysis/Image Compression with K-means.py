import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def kMeansInitialCentroids(X, K):
    randidx = range(X.shape[0])
    random.shuffle(randidx)
    centroids = X[randidx[0:K],:]
    return centroids

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

def runkMeans(X, initial_centroids, max_iters):
    centroids = initial_centroids
    for i in range(1,max_iters+1):
        idx = findCloestCentorids(X, centroids)
        centroids = computeCentroids(X, idx, K)

    return centroids,idx


A = mpimg.imread('/root/Machine Learning/ex7/bird_small.png')
#print A.shape

#plt.imshow(A)
#plt.axis('off')
#plt.show()

A = A / 255
img_size = A.shape
X = A.reshape(img_size[0]*img_size[1], 3)
m = X.shape[0]
n = X.shape[1]
K = 16
max_iters = 10
initial_centroids = kMeansInitialCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)
idx = np.array([int(x) for x in idx])
X_recovered  = centroids[idx-1,:]
X_recovered  = X_recovered.reshape((A.shape[0], A.shape[1], 3))
plt.subplot(1,2,1)
plt.title('Original')
plt.axis('off')
plt.imshow(A*255)
plt.subplot(1,2,2)
plt.title('Compressed, with %d colors'%K)
plt.axis('off')
plt.imshow(X_recovered*255)
plt.show()