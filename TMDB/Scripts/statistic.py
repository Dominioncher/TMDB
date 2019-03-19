from Content.Roots import OTHER_ROOT
from TMDB.Modules.Data import DataSet
from TMDB.Modules.Statistic.Correlations import correlation_table


# Вывод таблицы кореляций признаков
data_train = DataSet.read_train()
correlation_table(data_train, OTHER_ROOT + '/Correlations.png')
