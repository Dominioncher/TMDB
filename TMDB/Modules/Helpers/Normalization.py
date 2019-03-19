import pandas as pd
from sklearn import preprocessing


# Нормализация числового DataFrame
def normalize_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = frame.select_dtypes(include=[pd.np.number])
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    np_scaled = min_max_scaler.fit_transform(numeric_columns)
    df_normalized = pd.DataFrame(np_scaled, columns=numeric_columns.columns.values)
    frame.update(df_normalized)
    return frame
