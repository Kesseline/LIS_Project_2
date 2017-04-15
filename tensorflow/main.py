
import NNClassifier as Class
import numpy as np
import os

import crossval as cv

X = np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=(range(2, 17)))
y = np.loadtxt('train.csv', delimiter=',', skiprows=1, usecols=1, dtype='int')

y_alt = np.zeros((y.shape[0], 3))

for i in range(y.size):
    if y[i] == 0:
        y_alt[i][0] = 1
        y_alt[i][1] = 0
        y_alt[i][2] = 0
    elif y[i] == 1:
        y_alt[i][0] = 0
        y_alt[i][1] = 1
        y_alt[i][2] = 0
    else:
        y_alt[i][0] = 0
        y_alt[i][1] = 0
        y_alt[i][2] = 1

nn = Class.NNClassifier()

mcr = cv.validate(nn, X, y_alt)
print("loss: {}".format(mcr))
print("mcr: {}".format(np.mean(mcr)))

confirm = input("Would you like to apply the learner to the test data? (y/N)")
if (confirm == 'Y') | (confirm == 'y'):
    id_test = np.loadtxt('test.csv', delimiter=',', skiprows=1, usecols=0, dtype='int')
    X_test = np.loadtxt('test.csv', delimiter=',', skiprows=1, usecols=(range(1, 16)))

    y_fit = nn.fit(X, y_alt, X_test)

    y_fit = np.argmax(y_fit, 1)

    os.remove("out.csv")
    np.savetxt('out.csv', np.c_[id_test, y_fit], fmt="%d", header="Id,y", delimiter=",", comments="")
