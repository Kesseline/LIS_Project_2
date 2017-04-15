
import numpy as np


def validate(nn, X, y):
    k = 10
    mcr = np.zeros((k, 1))

    for l in range(k):
        (n, d) = X.shape

        x_train = []
        x_test = []
        y_train = []
        y_test = []

        for i in range(n-1):
            if i % k == l:
                x_test.append(X[i])
                y_test.append(y[i])
            else:
                x_train.append(X[i])
                y_train.append(y[i])

        x_train = np.array(x_train)
        x_test = np.array(x_test)
        y_train = np.array(y_train)
        y_test = np.array(y_test)

        y_fit = nn.fit(x_train, y_train, x_test)
        y_fit_alt = np.argmax(y_fit, 1)
        y_test_alt = np.argmax(y_test, 1)

        s = np.sum(y_fit_alt == y_test_alt)

        mcr[l] = 1 - (s / y_test.shape[0])

    return mcr
