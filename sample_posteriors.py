"""
File that draws samples from the posteriors of all of the supported models,
conditioned on all of the specified datasts.
"""
import random
from models import REGRESSION_MODEL_NAMES, sample_regression_model, \
                   CLASSIFICATION_MODEL_NAMES, sample_classification_model
from data.repo import get_downloaded_dataset_ids_by_task
from data.io import read_dataset_Xy, read_dataset_categorical, write_samples
from data.config import Preprocess

# For each different task (e.g. regression, classification, etc.),
# run outer loop that loops over datasets and inner loop that loops over models.

# choose different preprocessing for each fourth
# load num_categorical from raw data file

def sample_and_save_posteriors(dids, task):
    if task == 'regression':
        model_names = REGRESSION_MODEL_NAMES
        sample_model = sample_regression_model
    elif task == 'classification':
        model_names = REGRESSION_MODEL_NAMES
        sample_model = sample_classification_model
    else:
        raise ValueError('Invalid task: ' + task)
    
    num_datasets = len(dids)
    random.seed(12)
    random.shuffle(regression_dids)
    
    for i, dataset_id in enumerate(dids):
        # Partition datasets based on preprocessing
        if i < num_datasets * 1 // 4:
            preprocess = Preprocess.ONEHOT
        elif i < num_datasets * 2 // 4:
            preprocess = Preprocess.STANDARDIZED
        elif i < num_datasets * 3 // 4:
            preprocess = Preprocess.ROBUST
        else:
            preprocess = Preprocess.WHITENED
            
        # Suboptimal: this information could be moved
        # into the same file to make things slightly faster
        num_categorical = sum(read_dataset_categorical(dataset_id))
        X, y = read_dataset_Xy(dataset_id, preprocess)
        
        for model_name in model_names:
            samples = sample_model(model_name, X, y, num_categorical=num_categorical)
            write_samples(samples, model_name, dataset_id)


if __name__ == '__main__':
    # Get and shuffle dataset ids
    regression_dids = get_downloaded_dataset_ids_by_task('Supervised Regression')
    classification_dids = get_downloaded_dataset_ids_by_task('Supervised Classification')
    sample_and_save_posteriors(regression_dids, 'regression')
    sample_and_save_posteriors(classification_dids, 'classification')
