"""
Read all downloaded datasets, attempt to redownload any that fail to load,
and purge any that fail to redownload..
"""

import sys
import os
# must be run from the project root so the data package
# will be added to the path
sys.path.append(os.path.abspath('.'))

from data.io import get_downloaded_dataset_ids, read_dataset_and_purge


if __name__ == '__main__':
    dataset_ids = get_downloaded_dataset_ids()
    num_datasets = len(dataset_ids)
    start_i = 0
    for i in range(start_i, num_datasets):
        print('{} of {}'.format(i + 1, num_datasets), end='\t')
        dataset_id = dataset_ids [i]
        read_dataset_and_purge(dataset_id, verbose=True) 
