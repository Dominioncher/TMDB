import pandas as pd
from TMDB.Modules.Helpers.TolerantLabelEncoder import TolerantLabelEncoder


def simple_encoder_fit(train: pd.Series, test: pd.Series):
    le = TolerantLabelEncoder(ignore_unknown=True)
    train = le.fit_transform(train.astype(dtype=pd.np.str))
    test = le.transform(test.astype(dtype=pd.np.str))
    return train, test


def dummy_code_arrays(train: pd.Series, test: pd.Series):
    train = pd.get_dummies(train.apply(pd.Series).stack()).sum(level=0)
    test = pd.get_dummies(test.apply(pd.Series).stack()).sum(level=0)

    # Get missing columns in the training test
    missing_cols = set(train.columns) - set(test.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train.columns]
    return train, test
