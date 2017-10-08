"""
File for the IO operations used in the data package
"""

import os
import pickle
import pandas as pd

from .config import CONFIG, Preprocess

PICKLE_EXT = '.pkl'
OLD_PICKLE_EXT = '.pickle'
CSV_EXT = '.csv'
READING_ERRORS = EOFError
CSV_DEFAULT = True      # only for samples


def get_downloaded_dataset_ids(preprocess=Preprocess.RAW):
    """Get all dataset ids in the corresponding preprocessed data folder"""
    dataset_filenames = os.listdir(get_folder(preprocess))
    dataset_filenames.sort()
    return [int(filename.rstrip(OLD_PICKLE_EXT)) for filename in dataset_filenames]


def read_dataset(dataset_id, preprocess=Preprocess.RAW):
    """
    Read the dataset with specified preprocessing from disk and return the
    corresponding dictionary of its contents.
    """
    return read_file(get_dataset_filename(dataset_id, preprocess))
    
    
def read_task_dataset_ids(task):
    """Read dataset ids corresponding to specified task from disk"""
    return read_file(get_task_filename(task))


def read_samples(model_name, dataset_id, csv=CSV_DEFAULT):
    """Read model samples for specified dataset from disk"""
    filename = get_samples_filename(model_name, dataset_id, csv=csv)
    if csv:
        return pd.DataFrame.from_csv(filename)
    else:
        return read_file(filename)
    

def read_sample_diagnostics_list():
    """
    Return the sample diagnostics as a list where each element corresponds to
    the diagnostics of a single sampled posterior.
    """
    return list(read_sample_diagnostics_gen())


def read_sample_diagnostics_gen():
    """
    Returns a generator that reads the diagnostics of one sampled posterior at
    a time.
    """
    with open(CONFIG['diagnostics'], 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
    

def read_file(filename):
    """Read and return contents of file"""
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
    dataset_dict = read_dataset(dataset_id, preprocess)
    return dataset_dict['X'], dataset_dict['y']


# NOTE: This is used for the sampling methods that require the the number of
# categorical features. It is inefficient as this method also loads the raw
# dataset. The more proper thing to do if the preprocessing script were rerun
# would be to store the number of categorical features with each of the
# preprocessed datasetes.
def read_dataset_categorical(dataset_id):
    """Read the boolean list of which features are categorical from disk"""
    d = read_dataset(dataset_id, preprocess=Preprocess.RAW)
    return d['categorical']
    
    
def write_dataset(dataset_id, dataset, preprocess=Preprocess.RAW,
                  overwrite=True):
    """Write the dataset with specified preprocessing to disk"""
    write_dataset_dict(get_dataset_dict(dataset), dataset_id, preprocess,
                       overwrite)
        

def write_dataset_dict(d, dataset_id, preprocess=Preprocess.RAW,
                       overwrite=True):
    """Write the dataset dict with specified preprocessing to disk"""
    write_file(get_dataset_filename(dataset_id, preprocess), d, overwrite)
            
            
def write_task_dataset_ids(task, dids, overwrite=True):
    """Write dataset ids corresponding to specified task to disk"""
    write_file(get_task_filename(task), dids, overwrite)


def write_samples(samples_df, model_name, dataset_id, csv=CSV_DEFAULT,
                  overwrite=True):
    """Write model samples for specified dataset to disk"""
    filename = get_samples_filename(model_name, dataset_id, csv=csv)
    if csv:
        samples_df.to_csv(filename)
    else:
        samples_df.to_pickle(filename)
        
        
def append_sample_diagnostic(diagnostic):
    """
    Append object to diagnostics pickle file with protocol 2, so that it can be
    read by python2.
    """
    with open(CONFIG['diagnostics'], 'ab') as f:
        pickle.dump(diagnostic, f, protocol=2)

            
def write_file(filename, contents, overwrite=True):
    """Write file to disk if doesn't exist or overwrite is True"""
    if overwrite or not os.path.isfile(filename):
        with open(filename, 'wb') as f:
            pickle.dump(contents, f, protocol=pickle.HIGHEST_PROTOCOL)
        

def write_data_error(e, dataset_id, activity_type):
    """
    Append general dataset error information to corresponding file
    and add error type to set in corresponding pickle file. Abstaction of
    the 2 functions below
    """
    # Append to errors file
    filename = os.path.join(CONFIG['errors_folder'],
                            activity_type + '_errors.log')
    with open(filename, 'a') as f:
        print('Dataset id: {}\nError type: {}\nError message: {}'
                .format(dataset_id, type(e), str(e)), end='\n\n', file=f)
    
    # Update set with error type    
    filename = os.path.join(CONFIG['errors_folder'],
                            activity_type + '_error_set' + OLD_PICKLE_EXT)
    if not os.path.isfile(filename):
        error_set = set()
    else:
        with open(filename, 'rb') as f:
            error_set = pickle.load(f)
    error_set.add(type(e))
    with open(filename, 'wb') as f:
        pickle.dump(error_set, f)
    
    # Write error set to text file
    filename = os.path.join(CONFIG['errors_folder'],
                            activity_type + '_error_set.log')
    with open(filename, 'w') as f:
        print(str(error_set), file=f)

                    
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
     

def is_dataset_file(dataset_id, preprocess=Preprocess.RAW):
    """
    Return whether or not the dataset with specified preprocessing already
    exists on disk
    """
    return os.path.isfile(get_dataset_filename(dataset_id, preprocess))


def is_task_file(task):
    """
    Return whether or not the dataset ids corresponding to the specified task
    have already be downloaded
    """
    return os.path.isfile(get_task_filename(task))


def is_samples_file(model_name, dataset_id, csv=CSV_DEFAULT):
    """
    Return whether or not the samples file corresponding to the specified task
    model and dataset id is already on disk
    """
    return os.path.isfile(get_samples_filename(model_name, dataset_id, csv=csv))


def get_folder(preprocess=Preprocess.RAW):
    """Get folder of specified preprocessed data"""
    return CONFIG[preprocess.value + '_folder']
    
    
def get_dataset_filename(dataset_id, preprocess=Preprocess.RAW):
    """Get location of dataset"""
    return os.path.join(get_folder(preprocess), str(dataset_id) + OLD_PICKLE_EXT)


def get_task_filename(task):
    """Get location of dataset ids corresponding to specified task"""
    return os.path.join(CONFIG['tasks_folder'], task + OLD_PICKLE_EXT)


def get_samples_filename(model_name, dataset_id, csv=CSV_DEFAULT):
    """Get location of samples from specified model with specified dataset"""
    filename = '{}_{}'.format(dataset_id, model_name)
    if csv:
        filename = filename + CSV_EXT
    else:
        filename = filename + PICKLE_EXT
    return os.path.join(CONFIG['samples_folder'], filename)
   

# This is duplicated from the repo module because cyclic imports can be tricky,
# and I couldn't get it to work after much effort.
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
