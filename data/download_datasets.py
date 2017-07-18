"""
File for downloading the OpenML datasets
"""

import os
import pickle
import pandas as pd

import openml
from openml.exceptions import OpenMLServerError
from requests.exceptions import ChunkedEncodingError

OPENML_FOLDER = os.path.join(os.sep, *'/data/lisa/data/openml'.split('/'))
DATASETS_FOLDER = os.path.join(OPENML_FOLDER, 'datasets')
CHECKPOINT_ITERS = 25


def get_dataset_ids():
    """Get the ids of the dotasets to download"""
    dataset_metadata = openml.datasets.list_datasets()
    metadata_df = pd.DataFrame.from_dict(dataset_metadata, orient='index')
    filtered_df = metadata_df[metadata_df.NumberOfInstancesWithMissingValues == 0]
    return filtered_df.did.values
    
    
def download_datasets(dataset_ids, verbose=False):
    """Download the datasets that correspond to all of the give IDs"""
    good_dataset_ids = []
    bad_dataset_ids = []
    num_datasets = len(dataset_ids)
    exceptions = []
    for i, dataset_id in enumerate(dataset_ids):
        if verbose:
            print('{} of {}\tdataset ID: {} ...' \
                  .format(i, num_datasets, dataset_id), end=' ')
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            good_dataset_ids.append(dataset_id)
            write_dataset(dataset_id, dataset)
            if verbose: print('Success')
        # except (OpenMLServerError, ChunkedEncodingError) as e:
        except Exception as e:
            bad_dataset_ids.append(dataset_id)
            exceptions.append(e)
            if verbose: print('Failure')
        if (i + 1) % CHECKPOINT_ITERS == 0:
            if verbose:
                print('Reached iteration {}. Writing download info' \
                      .format(i + 1))
            write_download_info({
                'iteration': i + 1,
                'num_datasets': num_datasets,
                'good_dataset_ids': good_dataset_ids,
                'bad_dataset_ids': bad_dataset_ids,
                'exceptions': exceptions
            })

    
def write_download_info(info):
    """Write the information about the success/failure of downloading datasets"""
    filename = get_info_filename()
    with open(filename, 'wb') as f:
        pickle.dump(info, f) 


def get_info_filename():
    """Get location of where to write the download information"""
    return os.path.join(OPENML_FOLDER, 'info.pickle')


def get_dataset_filename(dataset_id):
    """Get location of where to write dataset"""
    return os.path.join(DATASETS_FOLDER, '{}.pickle'.format(dataset_id))


def write_dataset(dataset_id, dataset):
    """Write the dataset to disk"""
    filename = get_dataset_filename(dataset_id)
    with open(filename, 'wb') as f:
        pickle.dump(get_dataset_dict(dataset), f)
   
        
def get_dataset_dict(dataset):
    """Unpack the openml dataset object into a dictionary"""
    X, y, categorical, columns = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True,
        return_attribute_names=True)
    return {
        'X': X,
        'y': y,
        'categorical': categorical,
        'columns': columns
    }


if __name__ == '__main__':
    download_datasets(get_dataset_ids(), verbose=True)
    