"""
File that draws samples from the posteriors of all of the supported models,
conditioned on all of the specified datasts.
"""

import openml

from models import MODEL_NAMES, sample_model
from preprocessing.separation import separate_categorical

if __name__ == '__main__':
    dataset_ids = [23]
    
    # loop over datasets
    for dataset_id in dataset_ids:
        # download dataset and separate out features to use
        # TODO: write and change to function that loads dataset with options for
        # which features to use
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical, names = dataset.get_data(
            target=dataset.default_target_attribute,
            return_categorical_indicator=True,
            return_attribute_names=True)
        _, X_non_categ = separate_categorical(X, categorical)
        
        # loops over models
        for model_name in MODEL_NAMES:
            sample_model(model_name, X_non_categ, y)