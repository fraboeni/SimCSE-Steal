from datasets import load_dataset

"""
This file downloads the qqp dataset.
"""

qqp_train = load_dataset('merve/qqp',
                         split='train',
                         #download_mode='reuse_cache_if_exists',
                         cache_dir=".",
                         )



