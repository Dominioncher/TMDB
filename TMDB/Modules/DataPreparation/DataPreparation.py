import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# Подготовка данных + удаление ненужных колонок + добавление новых колонок
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    return data


# Заполнение пропущенных значений
def fill_na_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data


# Заполнение пропущенных значений для kaggle поскольку там нельзя выпиливать признаки
def fill_kaggle_na_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna(method='bfill')
    return data


# Разбиение выборки на тестовую и тренировочную
def data_split(data: pd.DataFrame, target_column_name: str):
    target = data[target_column_name]
    data = data.drop([target_column_name], axis=1)
    return train_test_split(data, target, test_size=0.33, random_state=42)


def fill_zero_genres(data: pd.DataFrame, genres)-> pd.DataFrame:
    zero_films = data[data['budget'] == 0]
    for i,row in zero_films.iterrows():
        budget = 0
        counter = 0
        for genre in genres:
            if row[genre] == 1:
                filled_budget = data[data[genre] == 1]
                budget = budget + filled_budget['budget'].mean()
                counter = counter + 1
        if budget == 0:
            data.drop([i], axis=0)
        else:
            data.set_value(i, 'budget', float(budget) / float(counter))

    return data

#Удаление ненужных
def drop_trash(data: pd.DataFrame)-> pd.DataFrame:
    return data.drop(["imdb_id", "original_title", "overview", "popularity", "poster_path", "status"], axis=1)
