import numpy as np
import cvxpy as cp


class KSVM(object):
    def __init__(self, kernel, kargs):
        # Kernel function
        self.kernel = kernel
        # Kernel function arguments
        self.kargs = kargs

    # Model training function
    def fit(self, X_train, Y_train, lamb):
        self.n = X_train.shape[0]
        self.X_train = X_train
        K_train = self.kernel(X_train, X_train, self.kargs)
        Y_train = Y_train.reshape(-1, 1).astype(np.float).T
        # Define minimization problem
        alpha = cp.Variable((self.n, 1))
        loss = cp.sum(cp.pos(1 - Y_train @ K_train @ alpha))
        reg = cp.norm(alpha.T @ K_train)
        #reg = cp.norm(cp.multiply(alpha, K_train))
        '''
        # Unsure about the regularization part.
        # Primal problem definition should be like
        psi = cp.Variable((1, self.n))
        loss = cp.sum(psi)
        reg = alpha.T @ K_train @ alpha
        constraints = [psi >= 0, Y_train @ K_train @ alpha.T + psi - 1 >= 0]
        prob = cp.Problem(cp.Minimize(loss / self.n + lamb * reg), constraints)
        '''
        prob = cp.Problem(cp.Minimize(loss / self.n + lamb * reg))
        prob.solve(verbose=True)
        self.alpha = np.array(alpha.value)
        self.alpha = self.alpha.T
        return self.predict_proba(X_train)

    # Label prediction function
    def predict(self, X_test):
        Y_pred, _ = self.predict_proba(X_test)
        return Y_pred

    # Label and probabilities prediction function
    def predict_proba(self, X_test):
        K_test = self.kernel(self.X_train, X_test, self.kargs)
        Y_proba = np.asarray(np.dot(self.alpha, K_test)).reshape(-1)
        Y_pred = np.sign(Y_proba).astype(int)
        return Y_pred, Y_proba