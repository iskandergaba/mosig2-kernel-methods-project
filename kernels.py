import os
import numpy as np
import scipy.sparse as sp
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
    Y_norms = X_norms if sym else np.mat(
        [np.mat(np.dot(v, v.T))[0, 0] for v in Y]).T

    K1 = X_norms * np.mat(np.ones((Y.shape[0], 1), dtype=np.float64)).T
    K2 = np.mat(np.ones((X.shape[0], 1), dtype=np.float64)) * Y_norms.T

    K = K1 + K2
    K -= 2 * np.mat(np.dot(X, Y.T))
    K *= -1. / (2 * np.power(sigma, 2))

    return np.exp(K)


# Compute the index of a k-mer in a sparse vector using an ordering
def _kmer_index(kmer, alpha):
    idx, k = 0, len(kmer)
    for i in range(k):
        letter = kmer[i]
        idx += alpha[letter] * len(alpha)**i
    return idx


# Find all possible variants of a k-mer with up to m mismatches
def _kmer_variants(kmer, alpha, m):
    k = len(kmer)
    letters = alpha.keys()
    variants = [kmer]
    for i in range(m):
        new_variants = []
        for var in variants:
            for j in range(k):
                for l in letters:
                    if l == var[j]:
                        continue
                    var_new = var[0:j] + l + var[j + 1:]
                    new_variants.append(var_new)
        variants.extend(new_variants)
    return variants


# Pre-indexation step for mismatch kernel
def _pre_mismatch(X, alpha, k, m):
    phis = sp.lil_matrix((X.shape[0], 4**k))
    for i in range(X.shape[0]):
        x = X[i][0]
        for j in range(len(x) - k + 1):
            kmer = x[j:j + k]
            variants = _kmer_variants(kmer, alpha, m)
            indices = [_kmer_index(var, alpha) for var in variants]
            for idx in indices:
                phis[i, idx] += 1
    return phis


def _mismatch(X, Y, args, phis_X=None):
    alpha, k, m = args[0], args[1], args[2]
    if phis_X is None:
        phis_X = _pre_mismatch(X, alpha, k, m)
    phis_Y = phis_X if isinstance(Y, type(None)) else _pre_mismatch(
        Y, alpha, k, m)
    return phis_X, phis_X.dot(phis_Y.T).toarray()


def mismatch(X, Y, args, phis=None):
    return _mismatch(X, Y, args, phis)


# k-Spectrum kernel is mismatch kernel with m = 0
def spectrum(X, Y, args, phis=None):
    args.append(0)
    return _mismatch(X, Y, args, phis)


# Combination of two k-Spectrum kernels with k1 and k2
# w1 is the weight for the k1-Spectrum
def spectrum_comb(X, Y, args, phis=None):
    alpha, k1, k2, w1 = args[0], args[1], args[2], args[3]
    phi_X, K1 = _mismatch(X, Y, [alpha, k1, 0], phis)
    _, K2 = _mismatch(X, Y, [alpha, k2, 0])
    return phi_X, w1 * K1 + (1 - w1) * K2


########################################
# START: LEGACY K-SPECTRUM KERNEL CODE #
########################################


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
                                                           kargs
                                                           for j in range(m)]
            K[i] = np.array(done + pool.map(kernel, inputs))
        else:
            inputs = [[X[i, 0], Y[j, 0]] + kargs for j in range(m)]
            K[i] = np.array(pool.map(kernel, inputs))

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


def spectrum_legacy(X, Y, args):
    return _string_kernel(X, Y, _spectrum, args)


########################################
# END: LEGACY K-SPECTRUM KERNEL CODE #
########################################
