import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from TMDB.Modules.DataPreparation.DataGeneration import parse_crew


# В ЭТОМ ФАЙЛЕ МЕТОДЫ НЕ ДОБАВЛЯЕМ ВЫНОСИТЕ ИХ В ОТДЕЛЬНЫЕ ФАЙЛЫ

# Выпиливание колонок

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop(["imdb_id", "original_title", "title", "overview", "poster_path", "status"], axis=1)
    return data


# добавление новых колонок
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data['genres'] = data['genres'].map(lambda x: [d['name'] for d in get_dictionary(x)])
    data['collection'] = data['belongs_to_collection'].map(lambda x: [d['name'] for d in get_dictionary(x)])
    data = data.drop(["belongs_to_collection"], axis=1)
    data['homepage'] = data['homepage'].notna()
    data['companies'] = data['production_companies'].map(lambda x: [d['name'] for d in get_dictionary(x)])
    data = data.drop(["production_companies"], axis=1)
    data['production_countries'] = data['production_countries'].map(lambda x: [d['name'] for d in get_dictionary(x)])
    data['spoken_languages'] = data['spoken_languages'].map(lambda x: [d['iso_639_1'] for d in get_dictionary(x)])
    data['Keywords'] = data['Keywords'].map(lambda x: [d['name'] for d in get_dictionary(x)])
    data['cast'] = data['cast'].map(lambda x: [d['name'] for d in get_dictionary(x)])
    data['crew_Directors'] = parse_crew(data, 'Director')
    data['crew_Producers'] = parse_crew(data, 'Producer')
    data['crew_Writers'] = parse_crew(data, 'Writer')
    data = data.drop(["crew"], axis=1)
    data = data.drop(["Keywords",'companies','cast', 'collection','homepage','crew_Directors', 'crew_Producers', 'crew_Writers', 'release_date', 'tagline'], axis=1)
    return data


# Заполнение пропущенных значений
def fill_na_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna(data.mean())
    return data


# Заполнение пропущенных значений для kaggle поскольку там нельзя выпиливать признаки
def fill_kaggle_na_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna(data.mean())
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
