import pandas as pd
from TMDB.Modules.Helpers.Result import result_frame
from TMDB.Modules.Models import Models as classifiers
from TMDB.Modules.Data import DataSet
from TMDB.Modules.DataPreparation import DataPreparation as preparation
from TMDB.Modules.Metrics import Metrics as metrics
from TMDB.Modules.Statistic.Describe import description

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('expand_frame_repr', False)

# Считали DataFrame c данными для обучения
data = DataSet.read_train()

# dummy кодирование жанра
data, genres = preparation.dummy_code_genres(data)
#data = preparation.dummy_code_production_companies(data)
# Фича инженеринг
data = preparation.feature_engineering(data)
# Заполнили пустые значения
data = preparation.fill_na_values(data)
#data = preparation.fill_zero_genres(data, genres.columns.values.tolist())
data = preparation.drop_zero_budget(data)
# Разделили на тестовую и обучающую выборки
data_train, data_test, target_train, target_test = preparation.data_split(data, 'revenue')

# Выбрали настроенный классификатор
estimator = classifiers.RandomForest

# Начали обучать его
print('start fit')
estimator.fit(data_train, target_train)
print('end fit')

# Предсказываем на которой учили
target_predict = estimator.predict(data_train)
print("On Train Set")
metrics.r2(target_train, target_predict)
# Статистика результата обучения
result = result_frame(target_train, target_predict)
print(description(result))

# Предсказываем на тестовой
target_predict = estimator.predict(data_test)
print("On Test Set")
metrics.r2(target_test, target_predict)
# Статистика результата обучения
result = result_frame(target_test, target_predict)
print(description(result))
