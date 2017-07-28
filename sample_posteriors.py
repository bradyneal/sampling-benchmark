"""
File that draws samples from the posteriors of all of the supported models,
conditioned on all of the specified datasts.
"""

import openml

from models import REGRESSION_MODEL_NAMES, sample_regression_model, \
                   CLASSIFICATION_MODEL_NAMES, sample_classification_model
from data import get_downloaded_dataset_ids_by_task, read_dataset_Xy, Preprocess

# For each different task (e.g. regression, classification, etc.),
# run outer loop that loops over datasets and inner loop that loops over models.

for dataset_id in get_downloaded_dataset_ids_by_task('Supervised Regression'):
    X, y = read_dataset_Xy(dataset_id, Preprocess.ONEHOT)
    for model_name in REGRESSION_MODEL_NAMES:
        sample_regression_model(model_name, X, y)
        
for dataset_id in get_downloaded_dataset_ids_by_task('Supervised Classification'):
    X, y = read_dataset_Xy(dataset_id, Preprocess.ONEHOT)
    for model_name in CLASSIFICATION_MODEL_NAMES:
        sample_classification_model(model_name, X, y)
