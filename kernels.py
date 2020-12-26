import numpy as np

def linear(X, Y):
    return np.dot(X, Y.T)

def polynomial(X, Y, d):
    return (np.identity + X.T * Y)**d

def rbf(X, Y, args):
    sigma = args[0]

    X_norms = np.mat([np.mat(np.dot(v, v.T))[0, 0] for v in X]).T
    Y_norms = np.mat([np.mat(np.dot(v, v.T))[0, 0] for v in Y]).T

    K1 = X_norms * np.mat(np.ones((Y.shape[0], 1), dtype=np.float64)).T
    K2 = np.mat(np.ones((X.shape[0], 1), dtype=np.float64)) * Y_norms.T

    K = K1 + K2
    K -= 2 * np.mat(np.dot(X, Y.T))
    K *= - 1./(2 * np.power(sigma, 2))

    return np.exp(K)
