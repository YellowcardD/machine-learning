import scipy.io as sio
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np

data = sio.loadmat('/root/Machine Learning/ex6/spamTrain.mat')

X = data['X']
y = data['y']
m = X.shape[0]
n = X.shape[1]
y = y.reshape((m,))

svm = SVC(C=0.1,kernel='linear')
svm.fit(X, y)
print (svm.score(X, y))

test = sio.loadmat('/root/Machine Learning/ex6/spamTest.mat')
X_test = test['Xtest']
y_test = test['ytest']
m = X_test.shape[0]
y_test = y_test.reshape((m,))
print (svm.score(X_test, y_test))