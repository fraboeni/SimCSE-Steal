import csv
import os

import pandas as pd

cwd = os.getcwd()
print('cwd: ', cwd)

test_data = pd.read_csv(
    './downloads/148fe59951311f5507e4d3f6ee80a0e392cc736800eb8a06baa3bfe3bc81d8de',
    sep=',', quoting=csv.QUOTE_ALL)
print('test data: ', test_data)

test_data_export = test_data[['question1', 'question2']]
test_data_export.to_csv('qqp_test.csv', index=False, quoting=csv.QUOTE_ALL, header=False)
print('exported test')

train_data = pd.read_csv(
    './downloads/480158dbb37ad1b381203a9fab5ea0859d072ed4a729712dcc05f52ec73a3136',
    sep=',', quoting=csv.QUOTE_ALL)
print('train data: ', train_data)

train_data_export = train_data[['question1', 'question2']]
train_data_export.to_csv('qqp_train.csv', index=False, quoting=csv.QUOTE_ALL, header=False)

print('exported train')
