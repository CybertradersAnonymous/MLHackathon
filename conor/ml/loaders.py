import pandas as pd

def load_train_nick():
    return pd.read_csv('../../data/train-nick.csv', index_col=0)

def load_train_tom():
    df = pd.read_csv('../../data/train-tom.csv', index_col=0, parse_dates=True)
    if df.index[0] > df.index[1]:
        df.sort_index(inplace=True)

    df.fillna(method='ffill', inplace=True)
    return df