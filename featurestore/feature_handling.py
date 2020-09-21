import pandas as pd


def save_csv(df, path):
    df.to_csv(path + '/store', index=False)


def load_csv(path, name):
    return pd.read_csv(path + '/' + name)
