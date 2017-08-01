"""
Read all downloaded datasets, attempt to redownload any that fail to load,
and purge any that fail to redownload.
"""

import sys
import os
# must be run from the project root so the data package
# will be added to the path
sys.path.append(os.path.abspath('.'))

from data.io import get_downloaded_dataset_ids, read_dataset, \
                    write_dataset_dict, delete_dataset, READING_ERRORS
from data.repo import download_dataset
from data.config import Preprocess


def purge_bad_datasets(start_i=0):
    dataset_ids = get_downloaded_dataset_ids()
    num_datasets = len(dataset_ids)
    for i in range(start_i, num_datasets):
        print('{} of {}'.format(i + 1, num_datasets), end='\t')
        dataset_id = dataset_ids [i]
        read_dataset_and_purge(dataset_id, verbose=True) 


def read_dataset_and_purge(dataset_id, verbose=False):
    """
    Read the raw dataset from disk and return the corresponding dictionary of
    its contents. If reading dataset errors, try to redownload the dataset. If
    that doesn't work, delete the dataset.
    """
    if verbose: print('Reading dataset {} ...'.format(dataset_id), end=' ')
    try:
        d = read_dataset(dataset_id, Preprocess.RAW)
        if verbose: print('Success!')
        return d
    except READING_ERRORS as e:
        if verbose: print('Failure!')
        if verbose: print('Downloading dataset {} ...'.format(dataset_id), end=' ')
        try:
            d = download_dataset(dataset_id)
            write_dataset_dict(d, dataset_id, Preprocess.RAW)
            if verbose: print('Success!')
        except Exception as e:
            if verbose: print('Failure! Deleting dataset {}'.format(dataset_id))
            delete_dataset(dataset_id, Preprocess.RAW)


if __name__ == '__main__':
    purge_bad_datasets()
