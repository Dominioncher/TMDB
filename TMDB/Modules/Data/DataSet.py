import pandas as pd
from Content.Roots import DATA_ROOT


def read_train() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT + "/train.csv", index_col=0)


def read_test() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT + "/test.csv", index_col=0)


def read_submission() -> pd.DataFrame:
    return pd.read_csv(DATA_ROOT + "/sample_submission.csv", index_col=0)

