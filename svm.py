import numpy as np
import cvxopt as cvx

'''
Much better, but still not working...
Web pages for help:
https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
https://medium.com/python-in-plain-english/introducing-python-package-cvxopt-implementing-svm-from-scratch-dc40dda1da1f
https://xavierbourretsicotte.github.io/SVM_implementation.html
'''

class KSVM(object):
    def __init__(self, kernel, kargs):
        # Kernel function
        self.kernel = kernel
        # Kernel function arguments
        self.kargs = kargs

    # Model training function
    def fit(self, X_train, Y_train, lamb=None, verbose=False):
        self.n = X_train.shape[0]
        self.X_train = X_train
        Y_train = Y_train.astype(np.float)

        K_train = self.kernel(X_train, X_train, self.kargs)
        #Y_K_train = Y_train * K_train
        #H = np.dot(Y_K_train.T, Y_K_train)
        H = np.outer(Y_train, Y_train) * K_train

        # Define minimization problem
        P = cvx.matrix(H)
        q = cvx.matrix(-np.ones(self.n))
        A = cvx.matrix(Y_train, (1, self.n))
        b = cvx.matrix(0.0)

        if lamb == None:
            G = cvx.matrix(np.diag(-np.ones(self.n)))
            h = cvx.matrix(np.zeros(self.n))
        else:
            G = cvx.matrix(
                np.vstack((np.diag(-np.ones(self.n)), np.identity(self.n))))
            h = cvx.matrix(
                np.hstack((np.zeros(self.n), lamb * np.ones(self.n))))

        # Set solver parameters
        cvx.solvers.options['show_progress'] = verbose
        '''
        cvx.solvers.options['abstol'] = 1e-10
        cvx.solvers.options['reltol'] = 1e-10
        cvx.solvers.options['feastol'] = 1e-10
        '''

        # Run solver
        solution = cvx.solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(solution['x'])
        self.alpha = self.alpha.T
        return self.predict_vals(X_train)

    # Label prediction function
    def predict(self, X_test):
        Y_pred, _ = self.predict_vals(X_test)
        return Y_pred

    # Label and probabilities prediction function
    def predict_vals(self, X_test):
        K_test = self.kernel(self.X_train, X_test, self.kargs)
        Y_vals = np.asarray(np.dot(self.alpha, K_test)).reshape(-1)
        Y_pred = np.sign(Y_vals).astype(int)
        return Y_pred, Y_vals
