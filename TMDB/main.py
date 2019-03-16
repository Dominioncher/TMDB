import pandas as pd 
from describe import print_description, file_description

submission_data = pd.read_csv("data/sample_submission.csv", index_col=0)
train_data = pd.read_csv("data/train.csv", index_col=0)
test_data = pd.read_csv("data/test.csv", index_col=0)

print_description(train_data)
file_description(train_data, 'output.txt')