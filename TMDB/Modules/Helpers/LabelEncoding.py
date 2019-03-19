from sklearn import preprocessing
import pandas as pd


# Кодирование не числовых объектов в уникальное число
def label_encode(data: pd.DataFrame) -> pd.DataFrame:
    le = preprocessing.LabelEncoder()
    for col in data.columns:
        if data[col].dtype == pd.np.object:
            data[col] = le.fit_transform(data[col].astype(dtype=pd.np.str))
    return data
