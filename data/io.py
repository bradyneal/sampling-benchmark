"""
File for the IO operations used in the data package
"""

import os
import pickle

from .config import CONFIG, Preprocess

PICKLE_EXT = '.pickle'


def get_downloaded_dataset_ids(preprocess=Preprocess.RAW):
    """Get all dataset ids in the corresponding preprocessed data folder"""
    dataset_filenames = os.listdir(get_folder(preprocess))
    return [int(filename.rstrip(PICKLE_EXT)) for filename in dataset_filenames]


def read_dataset_dict(dataset_id, preprocess=Preprocess.RAW):
    """Read the dataset with specified preprocessing from disk"""
    filename = get_dataset_filename(dataset_id, preprocess)
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def read_dataset_Xy(dataset_id, preprocess=Preprocess.RAW):
    """Read the dataset with specified preprocessing from disk"""
    dataset_dict = read_dataset_dict(dataset_id, preprocess)
    return dataset_dict['X'], dataset_dict['y']
    
    
def write_dataset(dataset_id, dataset, preprocess=Preprocess.RAW,
                  overwrite=False):
    """Write the dataset with specified preprocessing to disk"""
    write_dataset_dict(get_dataset_dict(dataset), dataset_id, preprocess,
                       overwrite)
        

def write_dataset_dict(d, dataset_id, preprocess=Preprocess.RAW,
                       overwrite=False):
    """Write the dataset dict with specified preprocessing to disk"""
    filename = get_dataset_filename(dataset_id, preprocess)
    if overwrite or not os.path.isfile(filename):
        with open(filename, 'wb') as f:
            pickle.dump(d, f)
            

def is_file(dataset_id, preprocess=Preprocess.RAW):
    """
    Return whether or not the dataset with specified preprocessing already
    exists on disk
    """
    filename = get_dataset_filename(dataset_id, preprocess)
    return os.path.isfile(filename)


def get_folder(preprocess=Preprocess.RAW):
    """Get folder of specified preprocessed data"""
    return CONFIG[preprocess.value + '_folder']
    
    
def get_dataset_filename(dataset_id, preprocess=Preprocess.RAW):
    """Get location of dataset"""
    return os.path.join(get_folder(preprocess), str(dataset_id) + PICKLE_EXT)
   
        
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
