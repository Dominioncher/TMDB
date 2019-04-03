import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# В ЭТОМ ФАЙЛЕ МЕТОДЫ НЕ ДОБАВЛЯЕМ ВЫНОСИТЕ ИХ В ОТДЕЛЬНЫЕ ФАЙЛЫ

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