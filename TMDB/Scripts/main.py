import pandas as pd
from TMDB.Modules.Helpers.Result import result_frame, kaggle_result
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
kaggle_test_data = DataSet.read_test()

# Фича инженеринг
data = preparation.feature_engineering(data)
# Заполнили пустые значения
data = preparation.fill_na_values(data)
# Заполнили пустые значения для kaggle
kaggle_test_data = preparation.fill_test_na_values(kaggle_test_data)
# Заполнение бюджета
data = preparation.drop_zero_budget(data)
# Кодирование строк
data, kaggle_test_data = preparation.label_coding(data, kaggle_test_data)

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











# Формируем решение для kaggle
target_predict = estimator.predict(kaggle_test_data)
kaggle_result(target_predict, kaggle_test_data)
