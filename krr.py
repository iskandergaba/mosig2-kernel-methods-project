import numpy as np
import kernels


class KernelRidgeRegression():

    def __init__(self, kernel, *kargs):
        self.kernel = kernel
        self.kargs = kargs

    # Function for model training
    def fit(self, X, Y, l):
        self.n = X.shape[0]
        self.K = self.kernel(X, self.kargs)
        self.alpha = np.dot(np.linalg.inv(self.K + l * np.identity(self.n, dtype=np.float)), Y)
        return self.predict_proba(X)

    def predict(self, X):
        Y_pred, _ = self.predict_proba(X)
        return Y_pred
    
    def predict_proba(self, X):
        Y_proba = np.empty((X.shape[0]), dtype=np.float)
        for i in range(0, X.shape[0]):
            Y_proba[i] = np.sum([np.dot(self.alpha[j], self.K[j, i]) for j in range(0, self.n)])
        Y_pred = np.round(Y_proba)
        return Y_pred.astype(int), Y_proba
