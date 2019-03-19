import pandas as pd
from Content.Roots import OTHER_ROOT
from TMDB.Modules.Data import DataSet
from TMDB.Modules.Statistic.Describe import description

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('expand_frame_repr', False)


# Описательная статистика сета
data_train = DataSet.read_train()
description(data_train, OTHER_ROOT + '/Describe.txt')
