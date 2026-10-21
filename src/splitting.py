"""Time-aware split: models learn from the past and are tested on the future."""
from . import config


def time_split(df, train_frac=config.TRAIN_FRAC, val_frac=config.VAL_FRAC):
    ts = df['timestamp']
    t_train = ts.quantile(train_frac)
    t_val = ts.quantile(train_frac + val_frac)
    train = df[ts <= t_train]
    val = df[(ts > t_train) & (ts <= t_val)]
    test = df[ts > t_val]
    if train.empty or val.empty or test.empty:
        raise ValueError('A split is empty (too many identical timestamps or too little data).')
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
