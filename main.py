import svm
import util
import kernels
import regression
import numpy as np
import pandas as pd
import encoder as enc
import preprocessor as pp

Xtrs = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
Xtrs_mat100 = ['data/Xtr0_mat100.csv', 'data/Xtr1_mat100.csv', 'data/Xtr2_mat100.csv']
Xtes = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
Xtes_mat100 = ['data/Xte0_mat100.csv', 'data/Xte1_mat100.csv', 'data/Xte2_mat100.csv']
Xtes_processed = [
    'data_processed/Xte0.csv', 'data_processed/Xte1.csv',
    'data_processed/Xte2.csv'
]
Ytrs = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']

label_code = {'A': '0.25', 'C': '0.5', 'G': '0.75', 'T': '1'}
alphabet = ['A', 'C', 'G', 'T']

def ksvm(X, Y, lamb, kernel, args, save_model=False, Xte=None):
    '''
    #pp.preprocess(Xtrs, Ytrs, Xtes, enc.one_hot_encode, enc_args=[alphabet])
    pp.preprocess(Xtrs, Ytrs, Xtes, enc.label_encode, enc_args=[label_code])
    # Load processed data
    X = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
    Y = pd.read_csv('data_processed/Ytr.csv', delimiter=',')
    X, Y = X.drop('Id', axis=1).to_numpy().astype(np.float), Y.drop(
        'Id', axis=1).to_numpy().reshape(-1)
    Y[Y == 0] = -1
    '''
    #X, Y = X[:2000], Y[:2000]
    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = util.train_test_split(X, Y, test_size=0.2)
    model = svm.KSVM(kernel, args)
    # Fit training data
    Y_pred, _ = model.fit(X_train, Y_train, lamb, verbose=True)
    acc = np.sum(Y_train == Y_pred) / Y_pred.shape[0]
    print('Accuracy over training data:', acc)
    # Test the model
    Y_pred, _ = model.predict_vals(X_val)
    acc = np.sum(Y_val == Y_pred) / Y_pred.shape[0]
    print('Accuracy over testing data:', acc)
    if save_model:
        if Xte is None:
            print("Please provide testing data set.")

        # Train new model on all the data
        model = svm.KSVM(kernel, args)
        Y_pred, _ = model.fit(X, Y, lamb)
        acc = np.sum(Y == Y_pred) / Y_pred.shape[0]
        print('Final model accuracy over training data:', acc)

        # Save test results
        Y_pred = model.predict(Xte).ravel()
        Y_pred[Y_pred == -1] = 0
        df = pd.DataFrame(data={'Bound': Y_pred})

        '''
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
        '''
        df.to_csv('data_processed/Yte.csv', index=True, index_label='Id')

def krr_numerical(X, Y, l, kernel, args, save_model=False, Xte=None):

    if save_model and Xte is None:
        print("Please provide a test dataframe to save predictions.")

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = util.train_test_split(X, Y, test_size=0.2)
    model = regression.KernelRidgeRegression(kernel, args)
    # Fit training data
    Y_pred, _ = model.fit(X_train, Y_train, l)
    acc = np.sum(Y_train == Y_pred) / Y_pred.shape[0]
    print('Accuracy over training data:', acc)
    # Test the model
    Y_pred, _ = model.predict_vals(X_val)
    test_acc = np.sum(Y_val == Y_pred) / Y_pred.shape[0]
    print('Accuracy over testing data:', test_acc)
    if save_model:
        # Train new model on all the data
        model = regression.KernelRidgeRegression(kernel, args)
        Y_pred, _ = model.fit(X, Y, l)
        acc = np.sum(Y == Y_pred) / Y_pred.shape[0]
        print('Final model accuracy over training data:', acc)

        Y_pred = model.predict(Xte).ravel()
        Y_pred[Y_pred == -1] = 0
        df = pd.DataFrame(data={'Bound': Y_pred})
        
        '''# Save test results
        df = pd.DataFrame() 
        for Xte in test_dir:
            data = pd.read_csv(Xte, delimiter=',')
            ids, seq = data['Id'], data.drop('Id', axis=1).to_numpy()
            Y_pred = model.predict(seq).ravel()
            Y_pred[Y_pred == -1] = 0
            temp = pd.DataFrame(data={'Id': ids, 'Bound': Y_pred})
            if df.empty:
                df = temp
            else:
                df = pd.concat([df, temp], axis=0)'''
        df.to_csv('data_processed/Yte.csv', index=True, index_label='Id')
    return test_acc


def krr_linear(X, Y, l, c=1, save_model=False, Xte=None):
    print('Linear Kernel Ridge Regression')
    krr_numerical(X, Y, l, kernels.linear, [c], save_model=save_model, Xte=Xte)


def krr_poly(X, Y, l, degree, c=1, gamma=1, save_model=False, Xte=None):
    print('Polynomial Kernel Ridge Regression')
    krr_numerical(X, Y, l,
                  kernels.polynomial, [degree, c, gamma],
                  save_model=save_model, Xte=Xte)


def krr_rbf(X, Y, l, sigma, save_model=False, Xte=None):
    print('RBF Kernel Ridge Regression')
    krr_numerical(X, Y, l, kernels.rbf, [sigma], save_model=save_model, Xte=Xte)


# The spectrum is not a numerical one. Will clean the code further later
def krr_spectrum(l, k, save_model=False):
    print('Spectrum Kernel Ridge Regression')
    pp.merge(Xtrs, 'data_processed/Xtr.csv')
    pp.merge(Ytrs, 'data_processed/Ytr.csv')

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
        Y_pred = model.predict(Xte).ravel()
        df = pd.DataFrame(data={'Bound': Y_pred})
        '''
        for Xte in Xtes_processed:
            data = pd.read_csv(Xte, delimiter=',')
            ids, seq = data['Id'], data.drop('Id', axis=1).to_numpy()
            Y_pred = model.predict(seq).ravel()
            temp = pd.DataFrame(data={'Id': ids, 'Bound': Y_pred})
            if df.empty:
                df = temp
            else:
                df = pd.concat([df, temp], axis=0)
        '''
        df.to_csv('data_processed/Yte.csv', index=True, index_label='Id')


# PREPROCESS THE DATA // TODO: POSSIBLY MOVE THIS INTO A FUNCTION

read_mat = True

if (not read_mat):
    # Either label encode
    #pp.preprocess(Xtrs, Ytrs, Xtes, enc.one_hot_encode, enc_args=[alphabet])
    pp.preprocess(Xtrs, Ytrs, Xtes, enc.label_encode, enc_args=[label_code])

# OR

# Read mat file
else:
    pp.merge(Xtrs_mat100, 'data_processed/Xtr.csv', delimiter=' ', save_index=True)
    pp.merge(Xtes_mat100, 'data_processed/Xte.csv', delimiter=' ', save_index=True)
    pp.merge(Ytrs, 'data_processed/Ytr.csv', read_header=0)

# Load processed data
X = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
Y = pd.read_csv('data_processed/Ytr.csv', delimiter=',')
Xte = pd.read_csv('data_processed/Xte.csv', delimiter=',')

X = X.drop('Id', axis=1).to_numpy().astype(np.float)
Xte = Xte.drop('Id', axis=1).to_numpy().astype(np.float)
Y = Y.drop('Id', axis=1).to_numpy().reshape(-1)
Y[Y == 0] = -1

#print(X)
#print(Y)
#print(Xte)
# Using our older RBF simply isn't working
#print('RBF Kernel SVM')
ksvm(X, Y, 100, kernels.rbf_svm, [5], save_model=True, Xte=Xte)

#krr_linear(X, Y, 0.01, 0.5, save_model=True, Xte=Xte)
#krr_poly(X, Y, 100, degree=2, c=0.1, gamma=0.5, save_model=True, Xte=Xte)
#krr_rbf(X, Y, 0.05, 7, save_model=True, Xte=Xte)

#lambdas = [1, 0.05]
#sigmas = [0.5, 0.05]
#
#for lam in lambdas:
#    for sigm in sigmas:
#        print("lambda", lam, "\tsigma", sigm)
#        krr_rbf(X, Y, lam, sigm)
#        print("\n\n")

#krr_rbf(X, Y, 0.5, 5, save_model=True, Xte=Xte)
#krr_spectrum(0.001, 8)
