import pandas as pd
from TMDB.Modules.Helpers.TolerantLabelEncoder import TolerantLabelEncoder


def simple_encoder_fit(train: pd.Series, test: pd.Series):
    le = TolerantLabelEncoder(ignore_unknown=True)
    train = le.fit_transform(train.astype(dtype=pd.np.str))
    test = le.transform(test.astype(dtype=pd.np.str))
    return train, test
