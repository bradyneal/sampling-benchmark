"""
File for the IO operations used in the data package
"""

import os
import pickle

from .config import CONFIG, Preprocess

PICKLE_EXT = '.pickle'
READING_ERRORS = EOFError


def get_downloaded_dataset_ids(preprocess=Preprocess.RAW):
    """Get all dataset ids in the corresponding preprocessed data folder"""
    dataset_filenames = os.listdir(get_folder(preprocess))
    return [int(filename.rstrip(PICKLE_EXT)) for filename in dataset_filenames]


def read_dataset(dataset_id, preprocess=Preprocess.RAW):
    """
    Read the dataset with specified preprocessing from disk and return the
    corresponding dictionary of its contents.
    """
    filename = get_dataset_filename(dataset_id, preprocess)
    with open(filename, 'rb') as f:
        return pickle.load(f)


def read_dataset_and_log(dataset_id, preprocess=Preprocess.RAW, verbose=False):
    """
    Read the dataset with specified preprocessing from disk and return the
    corresponding dictionary of its contents. Log any reading errors.
    """
    if verbose: print('Reading dataset {} ...'.format(dataset_id), end=' ')
    try:
        d = read_dataset(dataset_id, preprocess)
        if verbose: print('Success!')
        return d
    except READING_ERRORS as e:
        write_read_error(e, dataset_id)
        if verbose: print('Failure!')
        
    
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
       

def write_data_error(e, dataset_id, activity_type):
    """
    Append general dataset error information to corresponding file
    and add error type to set in corresponding pickle file. Abstaction of
    the 2 functions below
    """
    # Append to errors file
    filename = os.path.join(CONFIG['errors_folder'],
                            activity_type + '_errors.txt')
    with open(filename, 'a') as f:
        f.write('Dataset id: {}\nError type: {}\nError message: {}\n\n'
                .format(dataset_id, type(e), str(e)))
    
    # Update set with error type    
    filename = os.path.join(CONFIG['errors_folder'],
                            activity_type + '_error_set' + PICKLE_EXT)
    if not os.path.isfile(filename):
        error_set = set()
    else:
        with open(filename, 'rb') as f:
            error_set = pickle.load(f)
    error_set.add(type(e))
    with open(filename, 'wb') as f:
        pickle.dump(error_set, f)
    
    # Write error set to txt file
    filename = os.path.join(CONFIG['errors_folder'],
                            activity_type + '_error_set.txt')
    with open(filename, 'w') as f:
        pickle.dump(str(error_set), f)

                    
def write_read_error(e, dataset_id):
    """
    Write dataset read error information to corresponding file
    and add error type to set in corresponding pickle file
    """
    write_data_error(e, dataset_id, 'read')
            

def write_download_error(e, dataset_id):
    """
    Append dataset download error information to corresponding file
    and add error type to set in corresponding pickle file
    """
    write_data_error(e, dataset_id, 'download')
       
       
def delete_dataset(dataset_id, preprocess=Preprocess.RAW):
    """Remove the dataset with specified preprocessing from disk"""
    filename = get_dataset_filename(dataset_id, preprocess)
    os.remove(filename)
     

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
