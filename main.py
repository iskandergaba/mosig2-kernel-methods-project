import krr
import util
import kernels
import numpy as np
import pandas as pd
import preprocessor as pp

Xtrs = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
Ytrs = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']
Xtes = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
Xtes_processed = ['data_processed/Xte0.csv', 'data_processed/Xte1.csv', 'data_processed/Xte2.csv']
#label_code = {'A': '1', 'C': '1', 'G': '2', 'T': '3'}
label_code = {'A': '0.25', 'C': '0.5', 'G': '0.75', 'T': '1'}

pp.preprocess(Xtrs, Ytrs, Xtes, label_code)

# Load processed data
X = pd.read_csv('data_processed/Xtr.csv', delimiter=',')
Y = pd.read_csv('data_processed/Ytr.csv', delimiter=',')
X_ids, Y_ids = X['Id'], Y['Id']
X, Y = X.drop('Id', axis=1).to_numpy().astype(np.float), Y.drop('Id', axis=1).to_numpy().reshape(-1)

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = util.train_test_split(X, Y, test_size=0.2)

# Initialize model
model = krr.KernelRidgeRegression(kernels.rbf, [7])

# Fit training data
Y_pred, Y_proba = model.fit(X_train, Y_train, 0.04)
acc = np.sum(Y_train == Y_pred) / Y_pred.shape[0]
print('Accuracy over training data:', acc)

# Test the model
Y_pred, Y_proba = model.predict_proba(X_val)
acc = np.sum(Y_val == Y_pred) / Y_pred.shape[0]
print('Accuracy over testing data:', acc)

# Train new model on all the data
model = krr.KernelRidgeRegression(kernels.rbf, [7])
Y_pred, Y_proba = model.fit(X, Y, 0.04)
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
