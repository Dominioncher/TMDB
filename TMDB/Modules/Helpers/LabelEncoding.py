import pandas as pd
from TMDB.Modules.Helpers.TolerantLabelEncoder import TolerantLabelEncoder
import numpy as np


def simple_encoder_fit(train: pd.Series, test: pd.Series):
    le = TolerantLabelEncoder(ignore_unknown=True, unknown_encoded_value=0)
    train = le.fit_transform(train.astype(dtype=pd.np.str))
    test = le.transform(test.astype(dtype=pd.np.str))
    return train, test


def dummy_code(train: pd.Series, test: pd.Series):
    name = train.name
    train = pd.get_dummies(train.apply(pd.Series).stack(dropna=False)).sum(level=0)
    test = pd.get_dummies(test.apply(pd.Series).stack(dropna=False)).sum(level=0)

    train = train.rename(columns=lambda x: name + '_' + x)
    test = test.rename(columns=lambda x: name + '_' + x)

    # Get missing columns in the training test
    missing_cols = set(train.columns) - set(test.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train.columns]
    return train, test
