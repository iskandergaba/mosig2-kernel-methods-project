import os
import pandas as pd


def merge(Xs, save_filename, delimiter=',', read_header=None, save_index=False):

    df = pd.DataFrame()
    for X in Xs:
        temp0 = pd.read_csv(X, delimiter=delimiter, header=read_header)
        if df.empty:
            df = temp0
        else:
            df = pd.concat([df, temp0], axis=0)
    df.reset_index(drop=True, inplace=True)
    if save_index:
        df.to_csv(save_filename, sep=',', index=True, index_label='Id')
    else:
        df.to_csv(save_filename, sep=',', index=False)


def preprocess(Xtrs,
               Ytrs,
               Xtes,
               encoder,
               enc_args,
               save_path='data_processed'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Label-encode training data and merge training files
    df_train = pd.DataFrame()
    for Xtr in Xtrs:
        temp0 = pd.read_csv(Xtr, delimiter=',')
        temp1 = temp0['seq'].apply(lambda x: pd.Series(list(x))).apply(
            lambda x: encoder(x, enc_args))
        temp0 = pd.concat([temp0.Id, temp1], axis=1)
        if df_train.empty:
            df_train = temp0
        else:
            df_train = pd.concat([df_train, temp0], axis=0)
    df_train.reset_index(drop=True, inplace=True)
    df_train.to_csv(save_path + '/Xtr.csv', sep=',', index=False)

    # Merge training label files
    df_train_labels = pd.DataFrame()
    for Ytr in Ytrs:
        temp0 = pd.read_csv(Ytr, delimiter=',')
        if df_train_labels.empty:
            df_train_labels = temp0
        else:
            df_train_labels = pd.concat([df_train_labels, temp0], axis=0)
    df_train_labels.reset_index(drop=True, inplace=True)
    df_train_labels.to_csv(save_path + '/Ytr.csv', sep=',', index=False)

    # Label-encode testing data
    for Xte in Xtes:
        filename = Xte.split('/')[-1]
        df_test = pd.read_csv(Xte, delimiter=',')
        temp0 = df_test['seq'].apply(lambda x: pd.Series(list(x))).apply(
            lambda x: encoder(x, enc_args))
        df_test = pd.concat([df_test.Id, temp0], axis=1)
        df_test.reset_index(drop=True, inplace=True)
        df_test.to_csv(save_path + '/' + filename, sep=',', index=False)
