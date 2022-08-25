from datasets import load_dataset

"""
This file downloads the qqp dataset and preprocesses it into a CSV-file consisting only of two columns.
Each column contains one question.
"""

qqp_train = load_dataset('merve/qqp' ,split='train', download_mode='force_redownload', cache_dir=".")
#qqp_test = load_dataset('merve/qqp' ,split='test')


# remove unnecessary columns
df_qqp_train_q1q2 = qqp_train.remove_columns(["id", "qid1", "qid2", "is_duplicate"])
#df_qqp_test_q1q2 = qqp_test.remove_columns(["id", "qid1", "qid2", "is_duplicate"])


print("a")
df_qqp_train_q1q2.to_csv("qqp_train.csv")