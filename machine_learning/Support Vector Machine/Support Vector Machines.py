import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np


def plotData(X, y):
    pos = X[y==1]
    neg = X[y==0]
    plt.plot(pos[:,0], pos[:,1], 'k+', LineWidth=1, MarkerSize=7)
    plt.plot(neg[:,0], neg[:,1], 'ko', MarkerFaceColor='yellow', MarkerSize=7)
    #plt.xlim([0, 4.5])

def visualizeBoundaryLinear(X, y, coef, intercept):
    plotData(X, y)
    xp = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    yp = -(coef[0]*xp+intercept)/coef[1]
    plt.plot(xp, yp, '-b')
    plt.show()

def visualizeBoundary(X, y, model):

    plotData(X,y)
    x1 = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    x2 = np.linspace(min(X[:,1]), max(X[:,1]), 100)
    val = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            this_X = np.array([[x1[i], x2[j]]])
            val[i,j] = model.predict(this_X)

    val = val.T
    x1, x2 = np.meshgrid(x1, x2)
    plt.contour(x1, x2, val, [0.5], color='blue')
    plt.show()

def dataset3Params(X, y, X_val, y_val):

    minC  = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    minSigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    max = 0
    for i in range(len(minC)):
        for j  in range(len(minSigma)):
            clf = SVC(C=minC[i], gamma=1./2/minSigma[j]/minSigma[j], kernel='rbf')
            clf.fit(X, y)
            score = clf.score(X_val, y_val)
            print ('C = %f, sigma = %f, score = %f' %(minC[i], minSigma[j], score))
            if score>max:
                model = clf
                max = score
                C = minC[i]
                sigma = minSigma[j]

    return C, sigma, model

data1 = sio.loadmat('/root/Machine Learning/ex6/ex6data1.mat')

X = data1['X']
y = data1['y']
m = X.shape[0]
n = X.shape[1]
y = y.reshape((m,))

plotData(X, y)
plt.show()

C = [1, 100]
for i in C:
    clf1 = SVC(C=i, kernel='linear')
    clf1.fit(X, y)
    coef1 = (clf1.coef_).reshape((n,))
    intercept = clf1.intercept_
    visualizeBoundaryLinear(X, y, coef1, intercept)

data2 = sio.loadmat('/root/Machine Learning/ex6/ex6data2.mat')

X = data2['X']
y = data2['y']
m = X.shape[0]
n = X.shape[1]
y = y.reshape((m,))

plotData(X, y)
plt.show()

clf2 = SVC(gamma=100,kernel='rbf')
clf2.fit(X, y)
print (clf2.score(X, y))

visualizeBoundary(X, y, clf2)

data3 = sio.loadmat('/root/Machine Learning/ex6/ex6data3.mat')

X = data3['X']
y = data3['y']
X_val = data3['Xval']
y_val = data3['yval']
m = X.shape[0]
n = X.shape[1]
m_val = X_val.shape[0]
y = y.reshape((m,))

plotData(X, y)
plt.show()

C, sigma, model= dataset3Params(X, y, X_val, y_val)
print ('The optimal parameters is C = %f, sigma = %f' %(C, sigma))
visualizeBoundary(X, y, model)