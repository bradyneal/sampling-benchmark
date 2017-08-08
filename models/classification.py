"""
This module provides functions for sampling from the posteriors of various
classification models. The supported models are specified in the
CLASSIFICATION_MODEL_NAMES constant.
"""

import pymc3 as pm
import numpy as np
import theano.tensor as tt

from .utils import format_trace

CLASSIFICATION_MODEL_NAMES = ['softmax_linear']
NUM_SAMPLES = 500


def sample_classification_model(model_name, X, y, num_samples=NUM_SAMPLES,
                                num_non_categorical=None):
    """
    Sample from the posteriors of any of the supported models
    
    Args:
        model_name: to specify which model to sample from
        X: data matrix
        y: targets
        num_samples: number points to sample from the model posterior
    
    Returns:
        samples (in currently undecided format)
        
    Raises:
        ValueError: if the specified model name is not supported
    """
    if 'softmax_linear' == model_name:
        sample_softmax_linear(X, y, num_samples)
    else:
        raise ValueError('Unsupported model: {}\nSupported models: {}'
                         .format(model_name, CLASSIFICATION_MODEL_NAMES))

        
def sample_softmax_linear(X, y, num_samples=NUM_SAMPLES):
    """
    Sample from Bayesian Softmax Linear Regression
    """
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    Xt = theano.shared(X)
    with pm.Model() as model_softmax:
        W = pm.Normal('W', 0, sd=1e6, shape=(num_features, num_classes))
        b = pm.Flat('b', shape=num_classes)
        logit = Xt.dot(W) + b
        p = tt.nnet.softmax(logit)
        observed = pm.Categorical('obs', p=p, observed=y)
        trace = pm.sample(num_samples)
    return format_trace(trace)
