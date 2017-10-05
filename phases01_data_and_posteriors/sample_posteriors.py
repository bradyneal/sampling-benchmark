"""
File that draws samples from the posteriors of all of the supported models,
conditioned on all of the specified datasts.
"""
import random
from joblib import Parallel, delayed
from functools import partial
from traceback import print_exc

from models.regression import REGRESSION_MODEL_NAMES, sample_regression_model
from models.classification import CLASSIFICATION_MODEL_NAMES, sample_classification_model
from data.repo_local import get_downloaded_dataset_ids_by_task
from data.io import read_dataset_Xy, read_dataset_categorical, write_samples, \
                    is_samples_file, append_sample_diagnostic
from data.config import Preprocess
from data.preprocessing.format import to_ndarray

NUM_MODELS_PER_DATASET = 1
NUM_CORES_PER_CPU = 2
NUM_CPUS = 1
NUM_CORES = NUM_CPUS * NUM_CORES_PER_CPU
NUM_JOBS = int(NUM_CORES / 2)


# For each different task (e.g. regression, classification, etc.),
# run outer loop that loops over datasets and inner loop that loops over models.
def sample_and_save_posteriors(dids, task, seed=None):
    if task == 'regression':
        model_names = REGRESSION_MODEL_NAMES
        sample_model = sample_regression_model
    elif task == 'classification':
        model_names = CLASSIFICATION_MODEL_NAMES
        sample_model = sample_classification_model
    else:
        raise ValueError('Invalid task: ' + task)
    
    num_datasets = len(dids)
    if seed is not None:
        random.seed(seed)
    random.shuffle(dids)
    
    process_dataset_task = partial(process_dataset, model_names=model_names,
                                   sample_model=sample_model)
    Parallel(n_jobs=NUM_JOBS)(map(delayed(process_dataset_task), enumerate(dids)))


# def process_dataset(i, dataset_id):
def process_dataset(i_and_dataset_id, model_names, sample_model):
    """
    Sample from NUM_MODELS_PER_DATASET random model posteriors for the
    specified dataset. This function is run in parallel for many
    different datasets.
    """
    i, dataset_id = i_and_dataset_id
    # Partition datasets based on preprocessing
    if i % 3 == 0:
        preprocess = Preprocess.STANDARDIZED
    elif i % 3 == 1:
        preprocess = Preprocess.ROBUST
    elif i % 3 == 2:
        preprocess = Preprocess.WHITENED
        
    # Suboptimal: this information could be moved
    # into the same file to make things slightly faster
    num_non_categorical = read_dataset_categorical(dataset_id).count(False)
    X, y = read_dataset_Xy(dataset_id, preprocess)
    X = to_ndarray(X)
    
    random.shuffle(model_names)
    for model_name in model_names[:NUM_MODELS_PER_DATASET]:
        name = '{}_{}'.format(dataset_id, model_name)
        if is_samples_file(model_name, dataset_id):
            print(name + ' samples file already exists... skipping')
            continue                
        print('Starting sampling ' + name)
        try:
            output = sample_model(
                model_name, X, y, num_non_categorical=num_non_categorical)
            print('len(output):', len(output))
            samples, diagnostics = output
            # samples, diagnostics = sample_model(
            #     model_name, X, y, num_non_categorical=num_non_categorical)
            print('Finished sampling ' + name)
            if samples is not None:
                write_samples(samples, model_name, dataset_id, overwrite=False)
                if diagnostics is not None:
                    diagnostics['name'] = name
                    append_sample_diagnostic(diagnostics)
                else:
                    append_sample_diagnostic({'name': name, 'advi': True})
            else:
                print(name, 'exceeded hard time limit, so it was discarded')
        except Exception:
            print('Exception on {}:'.format(name))
            print_exc()


if __name__ == '__main__':
    # regression_dids = get_downloaded_dataset_ids_by_task('Supervised Regression')
    # sample_and_save_posteriors(regression_dids, 'regression')
    classification_dids = get_downloaded_dataset_ids_by_task('Supervised Classification')
    sample_and_save_posteriors(classification_dids, 'classification')
