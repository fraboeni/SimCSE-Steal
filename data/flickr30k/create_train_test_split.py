import pandas as pd
import csv


def main():
    df = pd.read_csv("flickr30k.csv", quoting=csv.QUOTE_ALL)
    total_size = df.shape[0] # total number of records: 317825

    # manually specify how much test data you want
    num_test = 30000 # 50000
    assert num_test % 10 == 0, "You need to specify multiples of 10 to prevent a mixing between training and test set."
    num_train = total_size-num_test

    # take the test samples from the beginning and train from end, so they should be disjoint, then shuffle them
    df_train = df.tail(num_train+1) # to encounter for starting at one in dataframes
    df_train = df_train.sample(frac=1)  # this is a hack to shuffle the dataframe

    df_test = df.head(num_test-1)
    df_test = df_test.sample(frac=1)  # this is a hack to shuffle the dataframe

    df_train.to_csv('flickr30k-train-samples-{}.csv'.format(num_train), index=False, quoting=csv.QUOTE_ALL, header=False)
    df_test.to_csv('flickr30k-test-samples-{}.csv'.format(num_test), index=False, quoting=csv.QUOTE_ALL, header=False)

if __name__ == '__main__':
    main()


