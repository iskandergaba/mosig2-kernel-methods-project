import util
import kernels
import numpy as np
import cvxpy as cp
import cvxopt as co
import pandas as pd

# lambda
l = 0.1

X = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
Y = pd.read_csv('data_processed/Ytr.csv', delimiter=',')
X, Y = X.drop('Id', axis=1).to_numpy().astype(np.float), Y.drop(
    'Id', axis=1).to_numpy().astype(np.float).reshape(-1, 1)
X_train, X_val, Y_train, Y_val = util.train_test_split(X, Y, test_size=0.2)
Y_train[Y_train == 0], Y_val[Y_val == 0] = -1, -1

n = X_train.shape[0]
K = kernels.rbf(X_train, X_train, [3])

alpha = cp.Variable((1,n))
psi = cp.Variable((1,n))

# This one takes forever
#objective = cp.Maximize(2 * np.dot(alpha.T, Y_train) - np.dot(np.dot(alpha.T, K), alpha))

#objective = cp.Minimize(1 / n * cp.multiply(alpha.T, Y_train) - cp.multiply(cp.multiply(alpha.T, K), alpha))

#beta = cp.Variable((n,1))
#v = cp.Variable()
#loss = cp.sum(cp.pos(1 - cp.multiply(Y, X @ beta - v)))
#reg = cp.norm(beta, 1)
#reg = cp.multiply(alpha.T, cp.multiply(K, alpha))
#reg = alpha @ K @ alpha.T
lambda_vals = [0.1, 1, 0.01]
loss = cp.sum(cp.pos(1 - Y_train.T @ K @ alpha.T))
reg = cp.norm(alpha @ K)
lambd = cp.Parameter(nonneg=True)
prob = cp.Problem(cp.Minimize(loss / n + l * reg))

for val in lambda_vals:
	lambd.value = val
	min_cost = prob.solve(verbose=True)
	value = np.array(alpha.value)
	print('Lambda =', val)
	print('Minimized cost:', min_cost)
	print('Alpha', value.shape)
	print(value)

#constraints = [alpha[:] * Y_train[:] >= 0]

'''
for i in range(n):
	constraints.append(alpha[i] * Y_train[i] >= 0)
	constraints.append(alpha[i] * Y_train[i] <= 1.0 / (2 * l * n))
'''

#prob = cp.Problem(objective, constraints)


'''
#alpha = cp.Variable((1, n), hermitian=True)
P = co.matrix(np.outer(Y_train,Y_train)*K)
q = co.matrix(np.ones(n)*-1)
A = co.matrix(Y_train,(n,1))
b = co.matrix(0.0)
G = co.matrix(np.diag(np.ones(n) * -1))
h = co.matrix(np.zeros(n))

print(K)

alphas = np.ravel(co.solvers.qp(P, q, G, h, A, b)['x'])
is_sv = alphas>1e-5

kernel = 'rbf'

_support_vectors = X_train[is_sv]
_n_support = np.sum(is_sv)
_alphas = alphas[is_sv]
_support_labels = Y_train[is_sv]
_indices = np.arange(n)[is_sv]
intercept = 0
for i in range(_alphas.shape[0]):
	intercept += _support_labels[i] 
	intercept -= np.sum(_alphas*_support_labels*K[_indices[i],is_sv])
intercept /= _alphas.shape[0]
weights = np.sum(X_train*Y_train.reshape(n,1)*_alphas.reshape(n,1),axis=0,keepdims=True) if kernel == "linear" else None

print(alphas)
'''
