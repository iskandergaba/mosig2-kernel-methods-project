import numpy as np


def train_test_split(X, Y, test_size=0.2):
    size = int(test_size * X.shape[0])
    all = np.arange(0, X.shape[0])
    test = np.random.choice(X.shape[0], size=size, replace=False)
    train = np.where(np.in1d(all, test, invert=True))[0]
    return X[train], X[test], Y[train], Y[test]