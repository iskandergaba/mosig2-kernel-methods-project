import os
import pandas as pd

def one_hot_encode(sequence, hot_code):
    for key, value in hot_code.items():
        sequence = sequence.replace(key, value)
    return sequence

def preprocess(Xtrs, Ytrs, Xtes, hot_code, save_path='data_processed'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # One-Hot encode training data and merge training files
    df_train = pd.DataFrame()
    for Xtr in Xtrs:
        temp0 = pd.read_csv(Xtr, delimiter=',')
        temp1 = temp0['seq'].apply(lambda x: one_hot_encode(x, hot_code)).apply(lambda x: pd.Series(list(x)))
        temp0 = pd.concat([temp0.Id, temp1], axis=1)
        if df_train.empty:
            df_train = temp0
        else:
            df_train = pd.concat([df_train, temp0], axis=0)
    df_train.reset_index(drop=True, inplace=True)
    df_train.to_csv(save_path + '/Xtr.csv', index=False)

    # Merge training label files
    df_train_labels = pd.DataFrame()
    for Ytr in Ytrs:
        temp0 = pd.read_csv(Ytr, delimiter=',')
        if df_train_labels.empty:
            df_train_labels = temp0
        else:
            df_train_labels = pd.concat([df_train_labels, temp0], axis=0)
    df_train_labels.reset_index(drop=True, inplace=True)
    df_train_labels.to_csv(save_path + '/Ytr.csv', index=False)

    # One-Hot encode testing data
    for Xte in Xtes:
        filename = Xte.split('/')[-1]
        df_test = pd.read_csv(Xte, delimiter=',')
        temp0 = df_test['seq'].apply(lambda x: one_hot_encode(x, hot_code)).apply(lambda x: pd.Series(list(x)))
        df_test = pd.concat([df_test.Id, temp0], axis=1)
        df_test.reset_index(drop=True, inplace=True)
        df_test.to_csv(save_path + '/' + filename, index=False)
