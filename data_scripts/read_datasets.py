"""
Read datasets, logging any errors.
"""

import sys
import os
# must be run from the project root so the data package
# will be added to the path
sys.path.append(os.path.abspath('.'))

from data.io import get_downloaded_dataset_ids, read_dataset_and_log
from data.config import Preprocess


def read_raw_downloaded_datasets(start_i=0):
    dataset_ids = get_downloaded_dataset_ids()
    num_datasets = len(dataset_ids)
    for i in range(start_i, num_datasets):
        print('{} of {}'.format(i + 1, num_datasets), end='\t')
        dataset_id = dataset_ids [i]
        read_dataset_and_log(dataset_id, verbose=True)
        

def read_all_datasets():
    dataset_ids = get_downloaded_dataset_ids()
    num_datasets = len(dataset_ids)
    for preprocess in Preprocess:
        print('### Reading {} preprocessed datasets ###'
              .format(preprocess.value))
        start_i = 0
        for i in range(start_i, num_datasets):
            print('{} of {}'.format(i + 1, num_datasets), end='\t')
            dataset_id = dataset_ids [i]
            read_dataset_and_log(dataset_id, preproces=Preprocess, verbose=True) 

if __name__ == '__main__':
    read_all_datasets()
