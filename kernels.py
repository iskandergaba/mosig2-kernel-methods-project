import numpy as np


def linear(X, Y, _):
    return np.dot(X, Y.T)


def polynomial(X, Y, args):
    d = args[0]
    return (np.identity + np.dot(X.T, Y))**d


def rbf(X, Y, args):
    sigma = args[0]

    X_norms = np.mat([np.mat(np.dot(v, v.T))[0, 0] for v in X]).T
    Y_norms = np.mat([np.mat(np.dot(v, v.T))[0, 0] for v in Y]).T

    K1 = X_norms * np.mat(np.ones((Y.shape[0], 1), dtype=np.float64)).T
    K2 = np.mat(np.ones((X.shape[0], 1), dtype=np.float64)) * Y_norms.T

    K = K1 + K2
    K -= 2 * np.mat(np.dot(X, Y.T))
    K *= -1. / (2 * np.power(sigma, 2))

    return np.exp(K)


def spectrum(X, Y, args):    
    def get_kernel(x, y, k):
        spec1 = np.array([x[i:i + k] for i in range(len(x) - k + 1)])
        spec2 = np.array([y[i:i + k] for i in range(len(y) - k + 1)])
        common = np.intersect1d(spec1, spec2)
        return np.sum(
            np.array([
                np.count_nonzero(spec1 == x) * np.count_nonzero(spec2 == x)
                for x in common
            ]))
    
    k = args[0]
    n, m = X.shape[0], Y.shape[0]
    kernel = np.empty(shape=(n, m), dtype=np.int)
    for i in range(n):
        for j in range(m):
            if i <= j or i >= m:
                kernel[i, j] = get_kernel(X[i], Y[j], k)
            else:
                kernel[i, j] = kernel[j, i]
    return kernel
