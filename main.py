import svm
import util
import kernels
import regression
import numpy as np
import pandas as pd
import encoder as enc
import preprocessor as pp

Xtrs = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
Ytrs = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']
Xtes = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
Xtes_processed = [
    'data_processed/Xte0.csv', 'data_processed/Xte1.csv',
    'data_processed/Xte2.csv'
]
label_code = {'A': '0.25', 'C': '0.5', 'G': '0.75', 'T': '1'}
alphabet = ['A', 'C', 'G', 'T']

def ksvm(lamb, kernel, args, save_model=False):
    #pp.preprocess(Xtrs, Ytrs, Xtes, enc.one_hot_encode, enc_args=[alphabet])
    pp.preprocess(Xtrs, Ytrs, Xtes, enc.label_encode, enc_args=[label_code])
    # Load processed data
    X = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
    Y = pd.read_csv('data_processed/Ytr.csv', delimiter=',')
    X, Y = X.drop('Id', axis=1).to_numpy().astype(np.float), Y.drop(
        'Id', axis=1).to_numpy().reshape(-1)
    Y[Y == 0] = -1
    #X, Y = X[:2000], Y[:2000]
    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = util.train_test_split(X, Y, test_size=0.2)
    model = svm.KSVM(kernel, args)
    # Fit training data
    Y_pred, y_proba = model.fit(X_train, Y_train, lamb)
    acc = np.sum(Y_train == Y_pred) / Y_pred.shape[0]
    print('Accuracy over training data:', acc)
    print(y_proba)
    print(np.where(y_proba < 0)[0].shape[0] + np.where(y_proba > 0)[0].shape[0])
    # Test the model
    Y_pred, _ = model.predict_proba(X_val)
    acc = np.sum(Y_val == Y_pred) / Y_pred.shape[0]
    print('Accuracy over testing data:', acc)
    if save_model:
        # Train new model on all the data
        model = svm.KSVM(kernel, args)
        Y_pred, _ = model.fit(X, Y, lamb)
        acc = np.sum(Y == Y_pred) / Y_pred.shape[0]
        print('Final model accuracy over training data:', acc)

        # Save test results
        df = pd.DataFrame()
        for Xte in Xtes_processed:
            data = pd.read_csv(Xte, delimiter=',')
            ids, seq = data['Id'], data.drop('Id', axis=1).to_numpy()
            Y_pred = model.predict(seq).ravel()
            Y_pred[Y_pred == -1] = 0
            temp = pd.DataFrame(data={'Id': ids, 'Bound': Y_pred})
            if df.empty:
                df = temp
            else:
                df = pd.concat([df, temp], axis=0)
        df.to_csv('data_processed/Yte.csv', index=False)

def krr_numerical(l, kernel, args, save_model=False):
    #pp.preprocess(Xtrs, Ytrs, Xtes, enc.one_hot_encode, enc_args=[alphabet])
    pp.preprocess(Xtrs, Ytrs, Xtes, enc.label_encode, enc_args=[label_code])
    # Load processed data
    X = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
    Y = pd.read_csv('data_processed/Ytr.csv', delimiter=',')
    X, Y = X.drop('Id', axis=1).to_numpy().astype(np.float), Y.drop(
        'Id', axis=1).to_numpy().reshape(-1)
    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = util.train_test_split(X, Y, test_size=0.2)
    model = regression.KernelRidgeRegression(kernel, args)
    # Fit training data
    Y_pred, _ = model.fit(X_train, Y_train, l)
    acc = np.sum(Y_train == Y_pred) / Y_pred.shape[0]
    print('Accuracy over training data:', acc)
    # Test the model
    Y_pred, _ = model.predict_proba(X_val)
    acc = np.sum(Y_val == Y_pred) / Y_pred.shape[0]
    print('Accuracy over testing data:', acc)
    if save_model:
        # Train new model on all the data
        model = regression.KernelRidgeRegression(kernel, args)
        Y_pred, _ = model.fit(X, Y, l)
        acc = np.sum(Y == Y_pred) / Y_pred.shape[0]
        print('Final model accuracy over training data:', acc)

        # Save test results
        df = pd.DataFrame()
        for Xte in Xtes_processed:
            data = pd.read_csv(Xte, delimiter=',')
            ids, seq = data['Id'], data.drop('Id', axis=1).to_numpy()
            Y_pred = model.predict(seq).ravel()
            temp = pd.DataFrame(data={'Id': ids, 'Bound': Y_pred})
            if df.empty:
                df = temp
            else:
                df = pd.concat([df, temp], axis=0)
        df.to_csv('data_processed/Yte.csv', index=False)


def krr_linear(l, c=1, save_model=False):
    print('Linear Kernel Ridge Regression')
    krr_numerical(l, kernels.linear, [c], save_model=save_model)


def krr_poly(l, degree, c=1, gamma=1, save_model=False):
    print('Polynomial Kernel Ridge Regression')
    krr_numerical(l,
                  kernels.polynomial, [degree, c, gamma],
                  save_model=save_model)


def krr_rbf(l, sigma, save_model=False):
    print('RBF Kernel Ridge Regression')
    krr_numerical(l, kernels.rbf, [sigma], save_model=save_model)


# The spectrum is not a numerical one. Will clean the code further later
def krr_spectrum(l, k, save_model=False):
    print('Spectrum Kernel Ridge Regression')
    pp.merge(Xtrs, Ytrs)

    # Load processed data
    X = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
    Y = pd.read_csv('data_processed/Ytr.csv', delimiter=',')
    X, Y = X.drop('Id',
                  axis=1).to_numpy(), Y.drop('Id',
                                             axis=1).to_numpy().reshape(-1)

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = util.train_test_split(X, Y, test_size=0.2)

    # Initialize model
    model = regression.KernelRidgeRegression(kernels.spectrum, [k])

    # Fit training data
    Y_pred, _ = model.fit(X_train, Y_train, l)
    acc = np.sum(Y_train == Y_pred) / Y_pred.shape[0]
    print('Accuracy over training data:', acc)

    # Test the model
    Y_pred, _ = model.predict_proba(X_val)
    acc = np.sum(Y_val == Y_pred) / Y_pred.shape[0]
    print('Accuracy over testing data:', acc)

    if save_model:
        # Train new model on all the data
        model = regression.KernelRidgeRegression(kernels.spectrum, [k])
        Y_pred, _ = model.fit(X, Y, l)
        acc = np.sum(Y == Y_pred) / Y_pred.shape[0]
        print('Final model accuracy over training data:', acc)

        # Save test results
        df = pd.DataFrame()
        for Xte in Xtes_processed:
            data = pd.read_csv(Xte, delimiter=',')
            ids, seq = data['Id'], data.drop('Id', axis=1).to_numpy()
            Y_pred = model.predict(seq).ravel()
            temp = pd.DataFrame(data={'Id': ids, 'Bound': Y_pred})
            if df.empty:
                df = temp
            else:
                df = pd.concat([df, temp], axis=0)
        df.to_csv('data_processed/Yte.csv', index=False)


print('RBF Kernel SVM')
ksvm(10, kernels.rbf, [7], save_model=True)
#ksvm(0.1, kernels.rbf, [7], save_model=True)
#krr_linear(0.01, 0.5, save_model=True)
#krr_poly(100, degree=2, c=0.1, gamma=0.5, save_model=True)
#krr_rbf(0.04, 7, save_model=True)
#krr_rbf(0.5, 5, save_model=True)
#krr_spectrum(0.001, 8)