import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# В ЭТОМ ФАЙЛЕ МЕТОДЫ НЕ ДОБАВЛЯЕМ ВЫНОСИТЕ ИХ В ОТДЕЛЬНЫЕ ФАЙЛЫ

# добавление новых колонок
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data['genres'] = data['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)]))

    return data


# Заполнение пропущенных значений
def fill_na_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data


# Заполнение пропущенных значений для kaggle поскольку там нельзя выпиливать признаки
def fill_kaggle_na_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna(method='bfill')
    return data


# Выпиливание колонок
def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(["imdb_id", "original_title", "overview", "popularity", "poster_path", "status", "genres"], axis=1)
    return data


# Разбиение выборки на тестовую и тренировочную
def data_split(data: pd.DataFrame, target_column_name: str):
    target = data[target_column_name]
    data = data.drop([target_column_name], axis=1)
    return train_test_split(data, target, test_size=0.33, random_state=42)


# Используется для парса Json с данными
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d
