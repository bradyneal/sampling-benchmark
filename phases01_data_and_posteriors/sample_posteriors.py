"""
File that draws samples from the posteriors of all of the supported models,
conditioned on all of the specified datasts.
"""
import random
from models.regression import REGRESSION_MODEL_NAMES, sample_regression_model
from models.classification import CLASSIFICATION_MODEL_NAMES, sample_classification_model
from data.repo_local import get_downloaded_dataset_ids_by_task
from data.io import read_dataset_Xy, read_dataset_categorical, write_samples, \
                    is_samples_file
from data.config import Preprocess
from data.preprocessing.format import to_ndarray

NUM_MODELS_PER_DATASET = 1


# For each different task (e.g. regression, classification, etc.),
# run outer loop that loops over datasets and inner loop that loops over models.
def sample_and_save_posteriors(dids, task):
    if task == 'regression':
        task_tag = 'regres'
        model_names = REGRESSION_MODEL_NAMES
        sample_model = sample_regression_model
    elif task == 'classification':
        task_tag = 'class'
        model_names = REGRESSION_MODEL_NAMES
        sample_model = sample_classification_model
    else:
        raise ValueError('Invalid task: ' + task)
    
    num_datasets = len(dids)
    random.seed(12)
    random.shuffle(regression_dids)
    
    for i, dataset_id in enumerate(dids):
        # Partition datasets based on preprocessing
        if i % 4 == 0:
            preprocess = Preprocess.ONEHOT
        elif i % 4 == 1:
            preprocess = Preprocess.STANDARDIZED
        elif i % 4 == 2:
            preprocess = Preprocess.ROBUST
        elif i % 4 == 3:
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
            samples = sample_model(model_name, X, y,
                                   num_non_categorical=num_non_categorical)
            print('Finished sampling ' + name)
            write_samples(samples, model_name, dataset_id, overwrite=False)


if __name__ == '__main__':
    regression_dids = get_downloaded_dataset_ids_by_task('Supervised Regression')
    sample_and_save_posteriors(regression_dids, 'regression')
    # classification_dids = get_downloaded_dataset_ids_by_task('Supervised Classification')
    # sample_and_save_posteriors(classification_dids, 'classification')