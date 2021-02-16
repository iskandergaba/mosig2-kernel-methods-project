import util
import preprocessor as pp
import main

alphabet = {'A': 0, 'C': 1, 'G': 2, 'T': 3}


# DATASET 0
Xtr, Ytr, Xte = main.preproess(0, numerical=False)
# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = util.train_test_split(Xtr,
                                                       Ytr,
                                                       test_size=0.2,
                                                       random=False)
main.krr_mismatch(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, alphabet, 0.3, 8, 1, 0, save_model=True)


# DATASET 1
Xtr, Ytr, Xte = main.preproess(1, numerical=False)
# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = util.train_test_split(Xtr,
                                                       Ytr,
                                                       test_size=0.2,
                                                       random=False)
main.krr_mismatch(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, alphabet, 0.6, 8, 1, 1, save_model=True)


# DATASET 2
Xtr, Ytr, Xte = main.preproess(2, numerical=False)
# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = util.train_test_split(Xtr,
                                                       Ytr,
                                                       test_size=0.2,
                                                       random=False)
main.krr_spectrum_comb(X_train, X_val, Y_train, Y_val, Xtr, Ytr, Xte, 0.23, 5, 7, 0.1111, 2, save_model=True)


# MERGE FILES INTO Yte.csv
Ytes = ['data_processed/Yte0.csv', 'data_processed/Yte1.csv', 'data_processed/Yte2.csv']
pp.merge(Ytes, 'Yte.csv', read_header=0, save_index=True)

