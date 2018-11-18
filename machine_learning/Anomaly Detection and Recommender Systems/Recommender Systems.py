import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.io as sio
from numpy import linalg
import math
from scipy.optimize import fmin_cg

def cofiCostFunc(parameters):
    X = parameters[0:num_movies*num_features].reshape((num_movies,num_features))
    Theta = parameters[num_movies*num_features:].reshape((num_users,num_features))
    M = np.dot(X, Theta.T)-Ynorm
    J =  0.5*sum(sum(R*(M**2)))+Lambda/2*sum(sum(Theta**2))+Lambda/2*sum(sum(X**2))
    return J

def cofiGrad(parameters):
    X = parameters[0:num_movies * num_features].reshape((num_movies, num_features))
    Theta = parameters[num_movies * num_features:].reshape((num_users, num_features))
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)
    for i in range(num_movies):
        ThetaTemp = Theta[R[i,:]==1,:]
        Ytemp = Ynorm[i,R[i,:]==1]
        X_grad[i,:] = np.dot(np.dot(X[i,:],ThetaTemp.T)-Ytemp,ThetaTemp)+Lambda*X[i,:]
    for j in range(num_users):
        Xtemp = X[R[:,j]==1,:]
        Ytemp = Ynorm[R[:,j]==1,j]
        Theta_grad[j,:] = np.dot((np.dot(Xtemp,Theta[j,:].T)-Ytemp).T,Xtemp)+Lambda*Theta[j,:]

    grad = np.append(X_grad.reshape((num_movies*num_features,)),Theta_grad.reshape((num_users*num_features,)))
    return grad

def normalizeRatings(Y, R):
    m, n = Y.shape
    Ymean = np.zeros((m,))
    Ynorm = np.zeros((Y.shape))
    for i in range(m):
        Ymean[i] = np.mean(Y[i, R[i,:]==1])
        Ynorm[i, R[i,:]==1] = Y[i, R[i,:]==1]-Ymean[i]

    return Ynorm, Ymean

data = sio.loadmat('/root/Machine Learning/ex8/ex8_movies.mat')
Y = data['Y']  #Y is a 1682*943 matrix, containing ratings(1-5) of 1682 movies on 943 users
R = data['R']  #R is a 1682*943 matrix, where R[i,j]=1 if and only if user j gave a rating to movie i

par = sio.loadmat('/root/Machine Learning/ex8/ex8_movieParams.mat')
X = par['X']
Theta = par['Theta']
num_users = 4
num_movies = 5
num_features = 3
Lambda = 1.5
X = X[0:num_movies,0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies,0:num_users]
R = R[0:num_movies,0:num_users]
params = np.append(X.reshape((num_movies*num_features,)),Theta.reshape((num_users*num_features,)))
#J = cofiCostFunc(params)
#G = cofiGrad(params)

Y = data['Y']
R = data['R']
Ynorm, Ymean = normalizeRatings(Y, R)
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10
X = np.random.randn(num_movies,num_features)
Theta = np.random.randn(num_users, num_features)
initial_parameters = np.append(X.reshape((num_movies*num_features,)),Theta.reshape((num_users*num_features,)))
Lambda = 10

theta = fmin_cg(cofiCostFunc,initial_parameters, fprime=cofiGrad, maxiter=100)
X = theta[0:num_movies*num_features].reshape((num_movies,num_features))
Theta = theta[num_movies*num_features:].reshape((num_users,num_features))
p = np.dot(X, Theta.T)
for i in range(p.shape[0]):
    p[i,:] = p[i,:]+Ymean[i]

print (p)
