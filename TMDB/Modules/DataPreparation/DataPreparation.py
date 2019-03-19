import pandas as pd
from sklearn.model_selection import train_test_split
from TMDB.Modules.Helpers.LabelEncoding import label_encode


# Подготовка данных + удаление ненужных колонок + добавление новых колонок
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    # data = data.drop(['name', 'ID', 'deadline', 'launched', 'currency', 'goal', 'pledged', 'usd pledged'], axis=1)
    # data = data.drop(['ID', 'goal', 'pledged', 'usd pledged'], axis=1)
    data = label_encode(data)
    return data


# Заполнение пропущенных значений
def fill_na_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data


# Разбиение выборки на тестовую и тренировочную
def data_split(data: pd.DataFrame, target_column_name: str):
    target = data[target_column_name]
    data = data.drop([target_column_name], axis=1)
    return train_test_split(data, target, test_size=0.33, random_state=42)


