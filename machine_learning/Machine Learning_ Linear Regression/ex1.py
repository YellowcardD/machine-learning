import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def PlotData(X, y):
    fig.add_subplot(1, 1, 1)
    plt.scatter(X, y, color='r',marker='x', s=50, label='Training data')
    plt.xlim(4, 25)
    plt.ylim(-5, 25)
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')


def computeCost(X, y, theta):
    return 1./2/m*sum((np.dot(X, theta)-y)**2)

def gradientDescent(X, y ,theta, alpha, num_iters):
    m = len(y) #number of trainging examples
    J_history = np.zeros((num_iters,1))
    for iter in np.arange(0,num_iters):
        theta = theta - alpha/m*np.dot(X.T, np.dot(X,theta)-y)
        J_history[iter] = computeCost(X, y, theta)
    return theta, J_history

data = pd.read_csv('/root/Machine Learning/ex1/ex1data1.txt', header=None)

fig = plt.figure()

X = data.ix[:,0]
X = np.array(X)

y = data.ix[:,1]
y= np.array(y)


m = len(y) #number of trainging examples

PlotData(X, y)

X = np.column_stack((np.ones((m,1)),X))
y = y.reshape(len(y),1)
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

theta, J_history = gradientDescent(X ,y ,theta, alpha, iterations)
print (theta)

plt.plot(X[:,1], np.dot(X,theta).reshape(m,), label='Linear regression')
plt.legend()
plt.show()

print ('For population = 35000, we predict a profit of ', np.dot([1, 3.5], theta)*10000)
print ('For population = 70000, we preidct a profit of ', np.dot([1, 7], theta)*10000)

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array((theta0_vals[i], theta1_vals[j])).reshape((2, 1))
        J_vals[i, j] = computeCost(X, y, t)

J_vals = J_vals.T
fig = plt.figure()
ax = Axes3D(fig)
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
ax.plot_surface(theta0_vals, theta1_vals, J_vals)
plt.show()

fig = plt.figure()
#theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
C = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20), color='black', linwidth=2)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.scatter(theta[0], theta[1], marker='x',color='r', s=50, linewidths=2)
plt.show()