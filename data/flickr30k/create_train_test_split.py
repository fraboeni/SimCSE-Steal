import pandas as pd
import csv


def main():
    df = pd.read_csv("flickr30k.csv")
    df_random = df.sample(frac=1) # this is a hack to shuffle the dataframe
    # total number of records: 317825
    total_size = df.shape[0]

    # manually specify how much test data you want
    num_test = 30000 # 50000
    num_train = total_size-num_test

    # take the train samples from the beginning and test from end, so they shoud be disjoint
    df_train = df_random.head(num_train)
    df_test = df_random.tail(num_test)

    df_train.to_csv('flickr30k-train-samples-{}.csv'.format(num_train), index=False, quoting=csv.QUOTE_ALL, header=False)
    df_test.to_csv('flickr30k-test-samples-{}.csv'.format(num_test), index=False, quoting=csv.QUOTE_ALL, header=False)

if __name__ == '__main__':
    main()


