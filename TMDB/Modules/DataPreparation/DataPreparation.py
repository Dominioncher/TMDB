import pandas as pd
from sklearn.model_selection import train_test_split
from TMDB.Modules.Helpers.LabelEncoding import label_encode


# Подготовка данных + удаление ненужных колонок + добавление новых колонок
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    # data = data.drop(['name', 'ID', 'deadline', 'launched', 'currency', 'goal', 'pledged', 'usd pledged'], axis=1)
    # data = data.drop(['ID', 'goal', 'pledged', 'usd pledged'], axis=1)
    data = label_encode(data)
    return data

def dummy_code_genres(data: pd.DataFrame):
    data['genres'] = data['genres'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
    genres = data.genres.str.get_dummies(sep=',')
    data = pd.concat([data, genres], axis=1, sort=False)
    data = data.drop(columns=["genres"])
    return data, genres.columns.values.tolist()

def dummy_code_production_companies(data: pd.DataFrame)-> pd.DataFrame:
    data['production_companies'] = data['production_companies'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
    companies = data.production_companies.str.get_dummies(sep=',')
    data = pd.concat([data, companies], axis=1, sort=False)
    data = data.drop(columns=["production_companies"])
    return data

def weight_code_production_companies(data: pd.DataFrame)-> pd.DataFrame:
    data['production_companies'] = data['production_companies'].map(lambda x: sorted([d['name'] for d in get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
    companies = data.production_companies.str.get_dummies(sep=',').columns.values.tolist()

    for i, row in data.iterrows():
        comp_names = row['production_companies'].split(sep=',')
        revenue = 0
        counter = 0
        for company in comp_names:
            filled_revenue = data[data['production_companies'].str.contains(company)]
            if filled_revenue['revenue'].mean() == filled_revenue['revenue'].mean():
                revenue = revenue + filled_revenue['revenue'].mean()
                counter = counter + 1
        if revenue == 0:
            data.drop([i], axis=0)
        else:
            data.set_value(i, 'company_weight', float(revenue) / float(counter))

    data['company_weight'] = (data['company_weight']-data['company_weight'].min())/(data['company_weight'].max()-data['company_weight'].min())
    data = data.drop(columns=['production_companies'])
    return data

def label_coding(data1: pd.DataFrame, data2: pd.DataFrame, ) -> pd.DataFrame:
#   data['genres'] = data['genres'].map(lambda x: sorted([d['name'] for d in preparation.get_dictionary(x)])).map(lambda x: ','.join(map(str, x)))
#   genres = data.genres.str.get_dummies(sep=',')
#   data = pd.concat([data, genres], axis=1, sort=False)
#   data = data.drop(columns=["genres"])
    data1 = label_encode(data1)
    data2 = label_encode(data2)
    return data1, data2

def fill_test_na_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.fillna(0)
    return data

# Заполнение пропущенных значений
def fill_na_values(data: pd.DataFrame) -> pd.DataFrame:
    data = data.dropna()
    return data

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

def drop_zero_budget(data: pd.DataFrame)-> pd.DataFrame:
    non_zero_films = data[data['budget'] != 0]
    return non_zero_films

# Разбиение выборки на тестовую и тренировочную
def data_split(data: pd.DataFrame, target_column_name: str):
    target = data[target_column_name]
    data = data.drop([target_column_name], axis=1)
    return train_test_split(data, target, test_size=0.33, random_state=42)

# Используется в разбиении объекта
def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d

#Удаление ненужных
def drop_trash(data: pd.DataFrame)-> pd.DataFrame:
    return data.drop(["imdb_id", "original_title", "overview", "popularity", "poster_path", "status"], axis=1)
