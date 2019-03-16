import pandas as pd 
import numpy as np

#data1 = pd.read_csv("data/sample_submission.csv", index_col=0)
data2 = pd.read_csv("data/train.csv", index_col=0)
#data3 = pd.read_csv("data/test.csv", index_col=0)

print(data2.info())
names = list()
for name, values in data2.iteritems():
    print(values.describe(), '\n')
    names.append(name)