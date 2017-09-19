"""
This module provides functions for sampling from the posteriors of various
classification models. The supported models are specified in the
CLASSIFICATION_MODEL_NAMES constant.
"""

import pymc3 as pm
import numpy as np
import theano.tensor as tt
from timeit import default_timer as timer

from .utils import format_trace
from .nn import sample_shallow_nn
from . import MAX_NUM_SAMPLES

# Arguably, build_pm_gp_cov should go in some 3rd file like util
from .regression import build_pm_gp_cov

CLASSIFICATION_MODEL_NAMES = \
    ['softmax_linear_class', 'shallow_nn_class',
     'gp_ExpQuad_class', 'gp_Exponential_class', 'gp_Matern32_class', 'gp_Matern52_class', 'gp_RatQuad_class']


def sample_classification_model(model_name, X, y, num_samples=MAX_NUM_SAMPLES,
                                num_non_categorical=None):
    """
    Sample from the posteriors of any of the supported models

    Args:
        model_name: to specify which model to sample from
        X: data matrix
        y: targets
        num_samples: number points to sample from the model posterior
        num_non_categorical: number of non-categorical features

    Returns:
        samples

    Raises:
        ValueError: if the specified model name is not supported
    """
    d = X.shape[1]
    X = reduce_data_dimension(X, model_name)
    reduced_d = X.shape[1]
    if reduced_d < d:
        num_non_categorical = reduced_d
    
    model_name = model_name.replace('_class', '')
    
    # Build model
    if 'softmax_linear' == model_name:
        model = build_softmax_linear(X, y)
    elif 'shallow_nn' == model_name:
        return sample_shallow_nn_class(X, y, num_samples)
    elif 'gp_ExpQuad' == model_name:
        model = build_gpc(X, y, 'ExpQuad')
    elif 'gp_Exponential' == model_name:
        model = build_gpc(X, y, 'Exponential')
    elif 'gp_Matern32' == model_name:
        model = build_gpc(X, y, 'Matern32')
    elif 'gp_Matern52' == model_name:
        model = build_gpc(X, y, 'Matern52')
    elif 'gp_RatQuad' == model_name:
        model = build_gpc(X, y, 'RatQuad')
    else:
        raise ValueError('Unsupported model: {}\nSupported models: {}'
                         .format(model_name, CLASSIFICATION_MODEL_NAMES))
    
    # Sample from model
    start = timer()
    with model:
        pm._log.info('Auto-assigning NUTS sampler...')
        if step is None:
            start_, step = pm.init_nuts(init='advi', njobs=1, n_init=200000,
                                        random_seed=-1, progressbar=True)
        
        for i, trace in enumerate(pm.iter_sample(MAX_NUM_SAMPLES, step)):
            elapsed = timer() - start
            if elapsed > MAX_TIME_IN_SECONDS:
                print('exceeded max time... breaking')
                break
    return format_trace(trace)


def model_softmax_linear(X, y):
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
    return model_softmax


def sample_shallow_nn_class(X, y, num_samples=MAX_NUM_SAMPLES):
    """
    Sample from shallow Bayesian neural network, using variational inference.
    Uses Categorical likelihood.
    """
    return sample_shallow_nn(X, y, 'classification')


def model_gpc(X, y, cov_f='ExpQuad'):
    """Sample from Gaussian Process"""
    # TODO also implement version that uses Elliptical slice sampling
    N, D = X.shape

    with pm.Model() as model_gp:
        # uninformative prior on the function variance
        log_s2_f = pm.Uniform('log_s2_f', lower=-10.0, upper=5.0)
        s2_f = pm.Deterministic('s2_f', tt.exp(log_s2_f))

        # covariance functions for the function f and the noise
        cov_func = s2_f * build_pm_gp_cov(D, cov_f)

        # Specify the GP.  The default mean function is `Zero`.
        f = pm.gp.GP('f', cov_func=cov_func, X=X, sigma=1e-6)
        # Smash to a probability
        f_transform = pm.invlogit(f)

        # Add the observations
        pm.Binomial('y', observed=y, n=np.ones(N), p=f_transform, shape=N)

    return model_gp
