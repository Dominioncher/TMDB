import pandas as pd 

submission_data = pd.read_csv("data/sample_submission.csv", index_col=0)
train_data = pd.read_csv("data/train.csv", index_col=0)
test_data = pd.read_csv("data/test.csv", index_col=0)


