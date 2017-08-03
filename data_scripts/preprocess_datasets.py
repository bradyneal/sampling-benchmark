"""
File for preprocessing the datasets
"""

import sys
import os
import scipy.sparse as sps
# must be run from the project root so the data package
# will be added to the path
sys.path.append(os.path.abspath('.'))

from data.config import Preprocess
from data.io import get_downloaded_dataset_ids, read_dataset_and_log, \
                    write_dataset_dict, is_file
from data.preprocessing import one_hot, standardize_and_one_hot, \
    robust_standardize_and_one_hot, whiten_and_one_hot, \
    to_ndarray, to_sparse


def preprocess_datasets(start_i=0, overwrite=False, verbose=True):
    dataset_ids = get_downloaded_dataset_ids(Preprocess.RAW)
    num_datasets = len(dataset_ids)
    for i in range(start_i, num_datasets):
        dataset_id = dataset_ids [i]
        if verbose:
            print('Preprocessing {} of {} (dataset_id: {})'
                  .format(i + 1, num_datasets, dataset_id))
        # Don't waste time reading dataset if it's already been preprocessed
        if is_file(dataset_id, Preprocess.WHITENED) and not overwrite:
            if verbose: print('Already preprocessed, so skipping')
            continue
        d = read_dataset_and_log(dataset_id, Preprocess.RAW)
        X, y, categorical = d['X'], d['y'], d['categorical']
        # convert to ndarray if not already
        X, _ = to_ndarray(X)
        write_preprocessed_dataset_dict(X, y, categorical, dataset_id,
                                        Preprocess.ONEHOT)
        write_preprocessed_dataset_dict(X, y, categorical, dataset_id,
                                        Preprocess.STANDARDIZED)
        write_preprocessed_dataset_dict(X, y, categorical, dataset_id,
                                        Preprocess.ROBUST)
        write_preprocessed_dataset_dict(X, y, categorical, dataset_id,
                                        Preprocess.WHITENED)


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
    
    preprocessed = to_sparse(preprocessed)
    write_dataset_dict({'X': preprocessed, 'y': y}, dataset_id, preprocess,
                       overwrite=True)
    

if __name__ == '__main__':
    preprocess_datasets(start_i=0, overwrite=True)
