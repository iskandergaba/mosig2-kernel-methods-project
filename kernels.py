import numpy as np

# Constructs the RBF Kernel Gram matrix K for the data input X
def rbf(X, sigma):
    X_norm = np.sum(X ** 2, axis = -1)
    return np.exp(-(1. / 2) * (X_norm[:,None] + X_norm[None,:] - 2 * np.dot(X, X.T)) / np.power(sigma, 2))
