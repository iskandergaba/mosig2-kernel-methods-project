import numpy as np


class KernelRidgeRegression():
    def __init__(self, kernel, kargs):
        # Kernel function
        self.kernel = kernel
        # Kernel function arguments
        self.kargs = kargs

    # Model training function
    def fit(self, X_train, Y_train, lamb):
        self.n = X_train.shape[0]
        self.X_train = X_train
        self.K_train = self.kernel(X_train, None, self.kargs)
        self.alpha = np.dot(
            np.linalg.inv(self.K_train +
                          lamb * self.n * np.identity(self.n, dtype=np.float)), Y_train)
        return self.predict_vals(X_train, self.K_train)

    # Label prediction function
    def predict(self, X_test, K_test=None):
        Y_pred, _ = self.predict_vals(X_test, K_test)
        return Y_pred

    # Label and values prediction function
    def predict_vals(self, X_test, K_test=None):
        if K_test is None:
            K_test = self.kernel(self.X_train, X_test, self.kargs)
        Y_vals = np.asarray(np.dot(self.alpha, K_test)).reshape(-1)
        Y_pred = np.sign(Y_vals).astype(int)
        return Y_pred, Y_vals
