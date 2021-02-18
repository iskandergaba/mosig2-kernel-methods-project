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
alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


def preproess(index, numerical=True, read_mat=False):

    Xtr, Xte = pd.DataFrame(), pd.DataFrame()
    if numerical:
        if read_mat:
            # READ MAT100 FILE AT INDEX index ONLY
            Xtr = pd.read_csv(Xtrs_mat100[index], delimiter=' ', header=None)
            Xte = pd.read_csv(Xtes_mat100[index], delimiter=' ', header=None)

        else:
            # label-encode
            pp.preprocess([Xtrs[index]], [Ytrs[index]], [Xtes[index]],
                          enc.label_encode,
                          enc_args=[label_code])
            # Load processed data
            Xtr = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
            Xtr = Xtr.drop('Id', axis=1).to_numpy().astype(np.float)

            Xte = pd.read_csv('data_processed/Xte.csv', delimiter=',')
            Xte = Xte.drop('Id', axis=1).to_numpy().astype(np.float)
    else:
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


def _krr_alphabetic(X_train,
                    X_val,
                    Y_train,
                    Y_val,
                    Xtr,
                    Ytr,
                    Xte,
                    lamb,
                    kernel,
                    args,
                    index,
                    save_model=False):
    # Initialize model
    model = regression.KernelRidgeRegression(kernel, args)

    if save_model:
        # Train new model on all the data
        model = regression.KernelRidgeRegression(kernel, args)
        Y_pred, _ = model.fit(Xtr, Ytr, lamb)
        acc = np.sum(Ytr == Y_pred) / Y_pred.shape[0]
        print('Final model accuracy over training data:', acc, '\n')

        # Save test results
        Y_pred = model.predict(Xte).ravel()
        Y_pred[Y_pred == -1] = 0
        df = pd.DataFrame(data={'Bound': Y_pred})
        df.to_csv('data_processed/Yte' + str(index) + '.csv',
                  index=True,
                  index_label='Id')

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


def krr_spectrum(X_train,
                 X_val,
                 Y_train,
                 Y_val,
                 Xtr,
                 Ytr,
                 Xte,
                 alpha,
                 lamb,
                 k,
                 index,
                 save_model=False):
    print('Spectrum Kernel Ridge Regression')
    return _krr_alphabetic(X_train,
                           X_val,
                           Y_train,
                           Y_val,
                           Xtr,
                           Ytr,
                           Xte,
                           lamb,
                           kernels.spectrum, [alpha, k],
                           index,
                           save_model=save_model)


def krr_spectrum_comb(X_train,
                      X_val,
                      Y_train,
                      Y_val,
                      Xtr,
                      Ytr,
                      Xte,
                      alpha,
                      lamb,
                      k1,
                      k2,
                      w1,
                      index,
                      save_model=False):
    print('Spectrum Combination Kernel Ridge Regression')
    return _krr_alphabetic(X_train,
                           X_val,
                           Y_train,
                           Y_val,
                           Xtr,
                           Ytr,
                           Xte,
                           lamb,
                           kernels.spectrum_comb, [alpha, k1, k2, w1],
                           index,
                           save_model=save_model)


def krr_mismatch(X_train,
                 X_val,
                 Y_train,
                 Y_val,
                 Xtr,
                 Ytr,
                 Xte,
                 alpha,
                 lamb,
                 k,
                 m,
                 index,
                 save_model=False):
    print('Mismatch Kernel Ridge Regression')
    return _krr_alphabetic(X_train,
                           X_val,
                           Y_train,
                           Y_val,
                           Xtr,
                           Ytr,
                           Xte,
                           lamb,
                           kernels.mismatch, [alpha, k, m],
                           index,
                           save_model=save_model)

def main():

    # Grid search, to be changed as needed
    sigmas = [0.005, 0.05, 0.5, 1, 3, 5, 7, 10]
    lambdas = np.linspace(0.01, 1, 100, endpoint=True)
    ws = np.linspace(0.01, 0.99, 10, endpoint=True)
    ks = [4, 5, 7, 8, 9, 10, 11, 12]
    ms = [0, 1, 2]

    best_params = [[], [], []]

    for index in range(0, 3):
        print("Dataset", index)
        best_acc = 0
        Xtr, Ytr, Xte = preproess(index, numerical=False)
        # Split the data into training and validation sets
        X_train, X_val, Y_train, Y_val = util.train_test_split(Xtr,
                                                               Ytr,
                                                               test_size=0.2,
                                                               random=False)

        for lamb in lambdas:
            #for sigma in sigmas:
            for w in ws:
                #print("Lambda = {0}, sigma = {1}".format(lamb, sigma))
                #print("Lambda = {0}, k = {1}".format(lamb, k))
                print("Lambda = {0}, w = {1}".format(lamb, w))
                #for m in ms:
                #print("Lambda = {0}, k = {1}, m = {2}".format(lamb, k, m))
                #acc = krr_rbf(lamb, sigma, save_model=False, read_mat=True)
                #acc = krr_spectrum(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, alphabet, lamb, k, index, save_model=False)
                acc = krr_spectrum_comb(X_train, X_val, Y_train, Y_val, Xtr,
                                        Ytr, Xte, alphabet, lamb, 5, 7, w,
                                        index)
                #acc = krr_mismatch(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, alphabet, lamb, k, m, index, save_model=False)
                print('\n')
                # Update the best accuracy and parameters
                if best_acc < acc:
                    best_acc = acc
                    #best_params[index] = [lamb, sigma]
                    best_params[index] = [lamb, w]
                    #best_params[index] = [lamb, k, m]

        #print("Best parameters:\nLambda = {0}\nSigma = {1}".format(best_params[index][0], best_params[index][1]))
        #krr_rbf(best_params[0], best_params[1], save_model=True, read_mat=True)
        #print("Best parameters:\nLambda = {0}, \nk = {1}, \nm = {2}".format(best_params[index][0], best_params[index][1], best_params[index][2]))
        #krr_mismatch(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, alphabet, best_params[index][0], best_params[index][1], best_params[index][2], index, save_model=True)
        #print("Best parameters:\nLambda = {0}\nK = {1}".format(best_params[index][0], best_params[index][1]))
        #krr_spectrum(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, alphabet, best_params[index][0], best_params[index][1], index, save_model=True)
        print("Best parameters:\nLambda = {0}\nK1 = {1}\nK2 = {2}\nw = {3}".
              format(best_params[index][0], 5, 7, best_params[index][1]))
        krr_spectrum_comb(X_train,
                          X_val,
                          Y_train,
                          Y_val,
                          Xtr,
                          Ytr,
                          Xte,
                          alphabet,
                          best_params[index][0],
                          5,
                          7,
                          best_params[index][1],
                          index,
                          save_model=True)

    Ytes = [
        'data_processed/Yte0.csv', 'data_processed/Yte1.csv',
        'data_processed/Yte2.csv'
    ]
    pp.merge(Ytes, 'data_processed/Yte.csv', read_header=0, save_index=True)


if __name__ == "__main__":
    main()
