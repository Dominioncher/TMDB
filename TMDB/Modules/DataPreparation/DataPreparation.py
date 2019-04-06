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

    def winter(x):
        if x == 12 or x < 3:
            return 1
        else:
            return 0

    def spring(x):
        if 2 < x < 6:
            return 1
        else:
            return 0

    def summer(x):
        if 5 < x < 9:
            return 1
        else:
            return 0

    def autumn(x):
        if 8 < x < 12:
            return 1
        else:
            return 0

    data['winter'] = data['release_date'].map(lambda x: 0 if pd.isna(x) else winter(int(x.split("/")[0])))
    data['spring'] = data['release_date'].map(lambda x: 0 if pd.isna(x) else spring(int(x.split("/")[0])))
    data['summer'] = data['release_date'].map(lambda x: 0 if pd.isna(x) else summer(int(x.split("/")[0])))
    data['autumn'] = data['release_date'].map(lambda x: 0 if pd.isna(x) else autumn(int(x.split("/")[0])))

    data['short_runtime'] = data['runtime'].map(lambda x: 1 if x < 60 else 0)
    data['usually_runtime'] = data['runtime'].map(lambda x: 1 if 60 <= x < 120 else 0)
    data['long_runtime'] = data['runtime'].map(lambda x: 1 if x >= 120 else 0)
    data = data.drop(
        ["Keywords", 'cast', 'collection', 'homepage', 'crew_Directors', 'crew_Producers', 'crew_Writers',
         'tagline'], axis=1)
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

def get_comp_weights(data:pd.DataFrame):
    companies_weight = dict()  # Словарь с весами компаний

    for i, row in data.iterrows():
        revenue = 0
        counter = 0
        for company in row['companies']:
            filled_revenue = data[data['companies'].apply(lambda x: company in x)]
            mean_revenue = filled_revenue['revenue'].dropna().mean()

            revenue += mean_revenue
            counter += 1

            companies_weight[company] = mean_revenue

    return companies_weight


def set_comp_weights(data:pd.DataFrame, company_weights:dict):
    for i, row in data.iterrows():
        revenue = 0
        counter = 0
        for company in row['companies']:
            if company not in company_weights:
                continue

            mean_revenue = company_weights[company]

            revenue = revenue + mean_revenue
            counter = counter + 1
        if counter == 0:
            data.set_value(i, 'company_weight', np.nan)
        else:
            data.set_value(i, 'company_weight', float(revenue) / float(counter))

    data = data.drop(columns=['companies'])

    data['company_weight'] = (data['company_weight'] - data['company_weight'].min()) / (
                data['company_weight'].max() - data['company_weight'].min())
    return data

def get_weights(data:pd.DataFrame, column:str):
    companies_weight = dict()  # Словарь с весами компаний

    for i, row in data.iterrows():
        revenue = 0
        counter = 0
        for col in row[column]:
            filled_revenue = data[data[column].apply(lambda x: col in x)]
            mean_revenue = filled_revenue['revenue'].dropna().mean()

            revenue += mean_revenue
            counter += 1

            companies_weight[col] = mean_revenue

    return companies_weight


def set_weights(data:pd.DataFrame, company_weights:dict):
    for i, row in data.iterrows():
        revenue = 0
        counter = 0
        for company in row['companies']:
            if company not in company_weights:
                continue

            mean_revenue = company_weights[company]

            revenue = revenue + mean_revenue
            counter = counter + 1
        if counter == 0:
            data.set_value(i, 'company_weight', np.nan)
        else:
            data.set_value(i, 'company_weight', float(revenue) / float(counter))

    data = data.drop(columns=['companies'])

    data['company_weight'] = (data['company_weight'] - data['company_weight'].min()) / (
                data['company_weight'].max() - data['company_weight'].min())
    return data

# Используется для парса Json с данными
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d
