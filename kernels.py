import os
import numpy as np
from multiprocessing import Pool


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


def _string_kernel(X, Y, kernel, kargs):
    n, m = X.shape[0], Y.shape[0]
    K = np.empty(shape=(n, m), dtype=np.float)
    for i in range(n):
        # Use the fact that K is symmetric
        done = [K[j, i] for j in range(i)] if i < m else []
        # Parallelize the inner loop
        pool = Pool(os.cpu_count())
        inputs = [[X[i, 0], Y[j, 0]] + kargs
                  for j in range(i, m)] if i < m else [[X[i, 0], Y[j, 0]] +
                                                       kargs for j in range(m)]
        K[i] = np.array(done + pool.map(kernel, inputs))
        # Scale-down the entries to the range [0, 1]
        K[i] = K[i] / np.max(np.max(K[i]), 0)
    return K


# Efficient k-spectrum implementation: https://dx.doi.org/10.1016/j.procs.2017.08.207
def _spectrum(args):
    x, y, k = args[0], args[1], args[2]
    phi, kmers = 0, {}
    for i in range(len(x) - k + 1):
        kmers[x[i:i + k]] = kmers[x[i:i + k]] + 1 if x[i:i + k] in kmers else 1
    for i in range(len(y) - k + 1):
        phi += kmers[y[i:i + k]] if y[i:i + k] in kmers else 0
    return phi


def spectrum(X, Y, args):
    return _string_kernel(X, Y, _spectrum, args)
