"""
File for downloading the OpenML datasets
"""

import os
import pickle
import pandas as pd

import openml
from openml.exceptions import OpenMLServerError, PyOpenMLError
from requests.exceptions import ChunkedEncodingError
from arff import BadNominalValue

# TODO elim repeat, move to config
UNIX_OPENML_PATH = '/data/lisa/data/openml'
OPENML_FOLDER = os.path.join(os.sep, *UNIX_OPENML_PATH.split('/'))
DATASETS_FOLDER = os.path.join(OPENML_FOLDER, 'datasets', 'raw')
CHECKPOINT_ITERS = 25


def get_dataset_ids():
    # TODO seems like repeat
    """Get the ids of the datasets to download"""
    dataset_metadata = openml.datasets.list_datasets()
    metadata_df = pd.DataFrame.from_dict(dataset_metadata, orient='index')
    filtered_df = metadata_df[metadata_df.NumberOfInstancesWithMissingValues == 0]
    # TODO explain type/dims being returned
    return filtered_df.did.values


def download_datasets(dataset_ids, start_iteration=0, verbose=False):
    # TODO explain start_iteration
    """Download the datasets that correspond to all of the give IDs"""
    num_datasets = len(dataset_ids)
    good_dataset_ids = []
    bad_dataset_ids = []
    exceptions = []

    # load previous saved values of above variables
    info_filename = get_info_filename()  # What is this??
    if os.path.isfile(info_filename):
        if verbose: print('Loading download info from file')
        with open(info_filename, 'rb') as f:
            info = pickle.load(f)
        start_iteration = info['iteration']
        good_dataset_ids = info['good_dataset_ids']
        bad_dataset_ids = info['bad_dataset_ids']
        exceptions = info['exceptions']

    # loop through dataset_ids and download corresponding datasets
    for i in range(start_iteration, num_datasets):
        dataset_id = dataset_ids[i]
        if verbose:
            print('{} of {}\tdataset ID: {} ...' \
                  .format(i + 1, num_datasets, dataset_id), end=' ')
        # OpenML likes to throw all kinds of errors when getting datasets
        try:
            dataset = openml.datasets.get_dataset(dataset_id)
            good_dataset_ids.append(dataset_id)
            write_dataset(dataset_id, dataset)
            if verbose: print('Success')
        # except (OpenMLServerError, PyOpenMLError, ChunkedEncodingError,
        #         BadNominalValue, EOFError) as e:
        except Exception as e:
            bad_dataset_ids.append(dataset_id)
            exceptions.append(e)
            if verbose: print('Failure')
        # checkpoint info
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
    # Maybe this should be global const??
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
