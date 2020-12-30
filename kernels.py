import numpy as np


def linear(X, Y, args):
    c = args[0]
    return np.dot(X, Y.T) + c


def polynomial(X, Y, args):
    d = args[0]
    c = args[1]
    gamma = args[2]
    return (gamma * np.dot(X, Y.T) + c)**d


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


def rbf_svm(X, Y, args=[5.0]):
    sigma = args[0]
    if (X.shape[0] == Y.shape[0]):
        return np.exp(-np.linalg.norm(X - Y)**2 / (2 * (sigma**2)))
    else:
        return rbf(X, Y, args)

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
