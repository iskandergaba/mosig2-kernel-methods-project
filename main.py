import svm
import util
import kernels
import regression
import numpy as np
import pandas as pd
import encoder as enc
import preprocessor as pp

Xtrs = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
Xtrs_mat100 = [
    'data/Xtr0_mat100.csv', 'data/Xtr1_mat100.csv', 'data/Xtr2_mat100.csv'
]
Xtes = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
Xtes_mat100 = [
    'data/Xte0_mat100.csv', 'data/Xte1_mat100.csv', 'data/Xte2_mat100.csv'
]
Xtes_processed = [
    'data_processed/Xte0.csv', 'data_processed/Xte1.csv',
    'data_processed/Xte2.csv'
]
Ytrs = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']

label_code = {'A': '0.25', 'C': '0.5', 'G': '0.75', 'T': '1'}
alphabet = ['A', 'C', 'G', 'T']


def preproess(index, numerical=True, read_mat=False):

    Xtr, Xte = pd.DataFrame(), pd.DataFrame()
    if numerical:
        if read_mat:
            # OLD CODE FOR MERGING
            '''
            # merge mat files in Xtr.csv, Xte.csv and Ytr.csv
            # add Id column (save_index) where necessary (the mat100 files don't have
            # it by default
            pp.merge(Xtrs_mat100,
                     'data_processed/Xtr.csv',
                     delimiter=' ',
                     save_index=True)
            pp.merge(Xtes_mat100,
                     'data_processed/Xte.csv',
                     delimiter=' ',
                     save_index=True)
            pp.merge(Ytrs, 'data_processed/Ytr.csv', read_header=0)
            '''

            # READ MAT100 FILE AT INDEX index ONLY
            Xtr = pd.read_csv(Xtrs_mat100[index], delimiter=' ', header=None)
            Xte = pd.read_csv(Xtes_mat100[index], delimiter=' ', header=None)

        else:
            # THIS IS NOT IDEAL
            # Use the same pp.preprocess calls but pass one file only

            # Either label encode or one-hot encode
            #pp.preprocess(Xtrs, Ytrs, Xtes, enc.one_hot_encode, enc_args=[alphabet])
            pp.preprocess([Xtrs[index]],
                          [Ytrs[index]],
                          [Xtes[index]],
                          enc.label_encode,
                          enc_args=[label_code])
            # Load processed data
            Xtr = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
            Xtr = Xtr.drop('Id', axis=1).to_numpy().astype(np.float)

            Xte = pd.read_csv('data_processed/Xte.csv', delimiter=',')
            Xte = Xte.drop('Id', axis=1).to_numpy().astype(np.float)
    else:
        # OLD CODE FOR MERGING
        '''
        pp.merge(Xtrs, 'data_processed/Xtr.csv', read_header=0)
        pp.merge(Xtes, 'data_processed/Xte.csv', read_header=0)
        pp.merge(Ytrs, 'data_processed/Ytr.csv', read_header=0)

        # Load processed data
        Xtr = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
        Xtr = Xtr.drop('Id', axis=1).to_numpy()

        Xte = pd.read_csv('data_processed/Xte.csv', delimiter=',')
        Xte = Xte.drop('Id', axis=1).to_numpy()
        '''

        # READ FILE AT INDEX index ONLY
        Xtr = pd.read_csv(Xtrs[index], delimiter=',', header=0)
        Xtr = Xtr.drop('Id', axis=1).to_numpy()

        Xte = pd.read_csv(Xtes[index], delimiter=',', header=0)
        Xte = Xte.drop('Id', axis=1).to_numpy()

    # Common part
    Ytr = pd.read_csv(Ytrs[index], delimiter=',', header=0)
    Ytr = Ytr.drop('Id', axis=1).to_numpy().reshape(-1)
    Ytr[Ytr == 0] = -1

    return Xtr, Ytr, Xte,


def ksvm(lamb, kernel, args, save_model=False, read_mat=False):
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
    X, Y, Xte = preproess(numerical=True, read_mat=read_mat)
    #X, Y = X[:2000], Y[:2000]
    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = util.train_test_split(X, Y, test_size=0.2)
    model = svm.KSVM(kernel, args)
    # Fit training data
    Y_pred = model.fit(X_train, Y_train, lamb, verbose=True)
    acc = np.sum(Y_train == Y_pred) / Y_pred.shape[0]
    print('Accuracy over training data:', acc)
    # Test the model
    Y_pred = model.predict(X_val)
    acc = np.sum(Y_val == Y_pred) / Y_pred.shape[0]
    print('Accuracy over testing data:', acc)
    if save_model:
        if Xte is None:
            print("Please provide testing data set.")

        # Train new model on all the data
        model = svm.KSVM(kernel, args)
        Y_pred = model.fit(X, Y, lamb)
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


def _krr_alphabetic(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, lamb, kernel, args, index, save_model=False):
    # Initialize model
    model = regression.KernelRidgeRegression(kernel, args)

    if save_model:
        # Train new model on all the data
        model = regression.KernelRidgeRegression(kernel, args)
        Y_pred, _ = model.fit(Xtr, Ytr, lamb)
        acc = np.sum(Ytr == Y_pred) / Y_pred.shape[0]
        print('Final model accuracy over training data:', acc)

        # Save test results
        Y_pred = model.predict(Xte).ravel()
        Y_pred[Y_pred == -1] = 0
        df = pd.DataFrame(data={'Bound': Y_pred})
        df.to_csv('data_processed/Yte'+index+'.csv', index=True, index_label='Id')

    else:
        # Fit training data
        Y_pred, _ = model.fit(X_train, Y_train, lamb)
        acc = np.sum(Y_train == Y_pred) / Y_pred.shape[0]
        print('Accuracy over training data:', acc)

        # Test the model
        Y_pred, _ = model.predict_vals(X_val)
        acc = np.sum(Y_val == Y_pred) / Y_pred.shape[0]
        print('Accuracy over testing data:', acc)
        return acc


def _krr_numerical(lamb, kernel, args, save_model=False, read_mat=False):

    Xtr, Ytr, Xte = preproess(numerical=True, read_mat=read_mat)

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = util.train_test_split(Xtr,
                                                           Ytr,
                                                           test_size=0.2)
    # Initialize the model
    model = regression.KernelRidgeRegression(kernel, args)
    # Fit training data
    Y_pred, _ = model.fit(X_train, Y_train, lamb)
    acc = np.sum(Y_train == Y_pred) / Y_pred.shape[0]
    print('Accuracy over training data:', acc)
    # Test the model
    Y_pred, _ = model.predict_vals(X_val)
    test_acc = np.sum(Y_val == Y_pred) / Y_pred.shape[0]
    print('Accuracy over testing data:', test_acc)
    if save_model:
        # Train new model on all the data
        model = regression.KernelRidgeRegression(kernel, args)
        Y_pred, _ = model.fit(Xtr, Ytr, lamb)
        acc = np.sum(Ytr == Y_pred) / Y_pred.shape[0]
        print('Final model accuracy over training data:', acc)

        Y_pred = model.predict(Xte).ravel()
        #Y_pred[Y_pred == -1] = 0
        df = pd.DataFrame(data={'Bound': Y_pred})
        df.to_csv('data_processed/Yte.csv', index=True, index_label='Id')
    return test_acc


def krr_linear(lamb, c=1, save_model=False, read_mat=False):
    print('Linear Kernel Ridge Regression')
    _krr_numerical(lamb,
                   kernels.linear, [c],
                   save_model=save_model,
                   read_mat=read_mat)


def krr_poly(lamb, degree, c=1, gamma=1, save_model=False, read_mat=False):
    print('Polynomial Kernel Ridge Regression')
    _krr_numerical(lamb,
                   kernels.polynomial, [degree, c, gamma],
                   save_model=save_model,
                   read_mat=read_mat)


def krr_rbf(lamb, sigma, save_model=False, read_mat=False):
    print('RBF Kernel Ridge Regression')
    return _krr_numerical(lamb,
                          kernels.rbf, [sigma],
                          save_model=save_model,
                          read_mat=read_mat)


def krr_spectrum(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, lamb, k, index, save_model=False):
    print('Spectrum Kernel Ridge Regression')
    _krr_alphabetic(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, lamb, kernels.spectrum, [k], index, save_model=save_model)


def krr_mismatch(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, lamb, k, m, index, save_model=False):
    print('Mismatch Kernel Ridge Regression')
    _krr_alphabetic(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, lamb, kernels.mismatch, [k, m], index, save_model=save_model)


# Using our older RBF simply isn't working
#print('RBF Kernel SVM')
#ksvm(100, kernels.rbf_svm, [5], save_model=True, read_mat=False)

# In case we want to conduct a grid search
sigmas = [0.005, 0.05, 0.5, 1, 3, 5, 7, 10]
lambdas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.1, 0.5, 1]
ks = [5, 6, 7, 8, 9]

trials, best_params = 1, [[],[],[]]



for index in range(3):
    print("Dataset", index)
    best_acc = 0
    Xtr, Ytr, Xte = preproess(index, numerical=False)
    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = util.train_test_split(Xtr,
                                                           Ytr,
                                                           test_size=0.2,
                                                           random=True)

    
    for lamb in lambdas:
        #for sigma in sigmas:
        for k in ks:
            #print("Lambda = {0}, sigma = {1}".format(lamb, sigma))
            print("Lambda = {0}, k = {1}".format(lamb, k))
            sum = 0
            for i in range(trials):
                print('Trial', i + 1)
                #sum += krr_rbf(lamb, sigma, save_model=False, read_mat=True)
                sum += krr_spectrum(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, lamb, k, index, save_model=False)
            acc = sum / trials
            print("Average testing accuracy: {0}\n".format(acc))
            # Update the best accuracy and parameters
            if best_acc < acc:
                best_acc = acc
                #best_params[i] = [lamb, sigma]
                best_params[i] = [lamb, k]

    print("Best parameters:\nLambda = {0}\nSigma = {1}".format(best_params[index][0], best_params[index][1]))
    #print("Best parameters:\nLambda = {0}\nK = {1}".format(best_params[index][0], best_params[index][1]))
    #krr_rbf(best_params[0], best_params[1], save_model=True, read_mat=True)
    krr_spectrum(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, best_params[index][0], best_params[index][1], index, save_model=True)


# Sample model calls
#krr_rbf(1, 0.05, save_model=True, read_mat=True)
#krr_linear(0.01, 0.5, save_model=True, read_mat=True)
#krr_poly(100, degree=2, c=0.1, gamma=0.5, save_model=True, read_mat=False)
#krr_spectrum(0.05, 8, save_model=False)
#krr_spectrum(0.1, 7, save_model=False)
#krr_spectrum(10000, 9, save_model=False)
#krr_mismatch(0.1, 12, 3, save_model=False)
