import numpy as np

class KernelRidgeRegression():

    def __init__(self, kernel, kargs):
        # Kernel function
        self.kernel = kernel
        # Kernel function arguments
        self.kargs = kargs

    # Model training function
    def fit(self, X_train, Y_train, l):
        self.n = X_train.shape[0]
        self.X_train = X_train
        self.K = self.kernel(X_train, X_train, self.kargs)
        self.alpha = np.dot(np.linalg.inv(self.K + l * np.identity(self.n, dtype=np.float)), Y_train)
        return self.predict_proba(X_train)
    
    # Label prediction function
    def predict(self, X_test):
        Y_pred, _ = self.predict_proba(X_test)
        return Y_pred
    
    # Label and probabilities prediction function
    def predict_proba(self, X_test):
        K_test = self.kernel(self.X_train, X_test, self.kargs)
        Y_proba = np.asarray(np.dot(self.alpha, K_test)).reshape(-1)
        Y_pred = np.round(Y_proba).astype(int)
        return Y_pred, Y_proba
