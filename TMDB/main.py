import pandas as pd 
import numpy as np 
import datetime

now = datetime.datetime.now()

data1 = pd.read_csv("sample_submission.csv", index_col=0) 
data2 = pd.read_csv("train.csv", index_col=0) 
data3 = pd.read_csv("test.csv", index_col=0) 

#print(data2.shape)
print(data2.info())
data2['new_date'] = pd.to_datetime(data2['release_date'], format='%m/%d/%y')
data2 = data2[data2.new_date > now]
#data2 = data2.drop(columns=['belongs_to_collection', 'genres', 'homepage', 'imdb_id', 'overview', 'popularity', 'poster_path', 'production_companies','production_countries', 'spoken_languages', 'status', 'Keywords', 'cast', 'crew'])
prep = data2['new_date']

q = prep.describe()
q1 = prep.unique().tolist()
q2 = prep.nonzero()[0].shape


names = list()
for name, values in data2.iteritems():
    print(values.describe())
    names.append(name)
