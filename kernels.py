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
    # Check if X=Y
    sym = isinstance(Y, type(None))
    if sym:
        Y = X

    sigma = args[0]

    X_norms = np.mat([np.mat(np.dot(v, v.T))[0, 0] for v in X]).T
    Y_norms = X_norms if sym else np.mat([np.mat(np.dot(v, v.T))[0, 0] for v in Y]).T

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
    # Check if X=Y
    sym = isinstance(Y, type(None))
    if sym:
        Y = X
    n, m = X.shape[0], Y.shape[0]
    K = np.empty(shape=(n, m), dtype=np.float)
    pool = Pool(os.cpu_count())
    for i in range(n):
        if sym:
            # Use the fact that K is symmetric
            done = [K[j, i] for j in range(i)] if i < m else []
            # Parallelize the inner loop
            inputs = [[X[i, 0], Y[j, 0]] + kargs
                      for j in range(i, m)] if i < m else [[X[i, 0], Y[j, 0]] +
                                                           kargs for j in range(m)]
            K[i] = np.array(done + pool.map(kernel, inputs))
            # Scale-down the entries to the range [0, 1]
            #K[i] = K[i] / max(np.max(K[i]), 1)
        else:
            inputs = [[X[i, 0], Y[j, 0]] + kargs for j in range(m)]
            K[i] = np.array(pool.map(kernel, inputs))

    # Or re-scale using the whole matrix
    #K /= np.max(K)
    return K


# Efficient k-spectrum implementation: https://dx.doi.org/10.1016/j.procs.2017.08.207
def _spectrum(args):
    x, y, k = args[0], args[1], args[2]
    # In Python, dictionaries act like hashtables
    phi, kmers = 0, {}
    for i in range(len(x) - k + 1):
        kmers[x[i:i + k]] = kmers[x[i:i + k]] + 1 if x[i:i + k] in kmers else 1
    for i in range(len(y) - k + 1):
        phi += kmers[y[i:i + k]] if y[i:i + k] in kmers else 0
    return phi

# Combination of two spectrum kernels with k1 and k2
# w1 is the weight for the k1-spectrum
def _spectrum_comb(args):
    x, y, k1, k2, w1 = args[0], args[1], args[2], args[3], args[4]
    return w1 * _spectrum([x, y, k1]) + (1 - w1) * _spectrum([x, y, k2])

# Very inefficient mismatch kernel implementation
def _mismatch(args):
    x, y, k, m = args[0], args[1], args[2], args[3]
    phi, kmers = 0, {}
    for i in range(len(x) - k + 1):
        kmers[x[i:i + k]] = kmers[x[i:i + k]] + 1 if x[i:i + k] in kmers else 1
    for i in range(len(y) - k + 1):
        #phi_part = 0
        for key in kmers:
            diff = sum(c1 != c2 for c1, c2 in zip(key, y[i:i + k]))
            #phi_part += kmers[key] if kmers[key] > phi_part and diff <= m else phi_part
            phi += kmers[key] if diff <= m else 0
        #phi += phi_part
    return phi

def _gap_weighted(args):
    x, y, p, lamK = args[0], args[1], args[2], args[3]
    n = len(x)
    m = len(y)
    DPS = np.zeros([n+1, m+1])
    DP = np.zeros([n+1, m+1])
    for i in range(0, n):
        for j in range(0, m):
            if x[i] == y[j]:
                DPS[i+1, j+1] = lamK**2
    for l in range(2, p+1):
        kern = 0
        for i in range(1, n):
            for j in range(1, m):
                DP[i, j] = DPS[i, j] + lamK * DP[i-1, j] + lamK * DP[i, j-1] - lamK**2 * DP[i-1, j-1]
                if x[i-1] == y[j-1]:
                    DPS[i, j] = lamK**2 * DP[i-1, j-1]
                    kern = kern + DPS[i, j]
    return kern

def spectrum(X, Y, args):
    return _string_kernel(X, Y, _spectrum, args)

def spectrum_comb(X, Y, args):
    return _string_kernel(X, Y, _spectrum_comb, args)

def mismatch(X, Y, args):
    return _string_kernel(X, Y, _mismatch, args)

def gap_weighted(X, Y, args):
    return _string_kernel(X, Y, _gap_weighted, args)
