"""
This module provides functions for sampling from the posteriors of various
classification models. The supported models are specified in the
CLASSIFICATION_MODEL_NAMES constant.
"""

import pymc3 as pm

CLASSIFICATION_MODEL_NAMES = []


def sample_classification_model(model_name, X, y, num_samples=NUM_SAMPLES,
                                num_non_categorical=None):
    pass
