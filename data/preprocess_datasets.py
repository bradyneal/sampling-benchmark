"""
File for preprocessing the datasets
"""

import sys
import os
# must be run from the project root so the data package
# will be added to the path
sys.path.append(os.path.abspath('.'))

from data.config import Preprocess
from data.io import get_all_dataset_ids, read_dataset, write_dataset_dict
from data.preprocessing import one_hot, standardize_and_one_hot, \
    robust_standardize_and_one_hot, whiten_and_one_hot


def preprocess_datasets(verbose=False):
    dataset_ids = get_all_dataset_ids(Preprocess.RAW)
    num_datasets = len(dataset_ids)
    for i, dataset_id in enumerate(dataset_ids):
        if verbose:
            print('Preprocessing {} of {} (dataset_id: {})'
                  .format(i + 1, num_datasets, dataset_id), end=' ')
        d = read_dataset(dataset_id, Preprocess.RAW)
        if verbose: print('.', end='')
        X, y, categorical = d['X'], d['y'], d['categorical']
        write_preprocessed_dataset_dict(X, y, categorical, dataset_id,
                                        Preprocess.ONEHOT)
        if verbose: print('.', end='')
        write_preprocessed_dataset_dict(X, y, categorical, dataset_id,
                                        Preprocess.STANDARDIZED)
        if verbose: print('.', end='')
        write_preprocessed_dataset_dict(X, y, categorical, dataset_id,
                                        Preprocess.ROBUST)
        if verbose: print('.', end='')
        write_preprocessed_dataset_dict(X, y, categorical, dataset_id,
                                        Preprocess.WHITENED)
        if verbose: print('.', end='\n')


def write_preprocessed_dataset_dict(X, y, categorical, dataset_id, preprocess):
    if Preprocess.RAW is preprocess:
        preprocessed = X
    elif Preprocess.ONEHOT is preprocess:
        preprocessed = one_hot(X, categorical)
    elif Preprocess.STANDARDIZED is preprocess:
        preprocessed = standardize_and_one_hot(X, categorical)
    elif Preprocess.ROBUST is preprocess:
        preprocessed = robust_standardize_and_one_hot(X, categorical)
    elif Preprocess.WHITENED is preprocess:
        preprocessed = whiten_and_one_hot(X, categorical)
    else:
        raise ValueError('Unsupported preprocessing type: {}'.format(preprocess))
    write_dataset_dict({'X': preprocessed, 'y': y}, dataset_id, preprocess)
    

if __name__ == '__main__':
    preprocess_datasets(verbose=True)
