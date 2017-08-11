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
    elif 'shallow_nn' == model_name:
        sample_shallow_nn(X, y, num_samples)
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


def sample_shallow_nn(X, y, num_samples=NUM_SAMPLES, num_hidden=100, vi=True):
    """
    Sample from shallow Bayesian neural network
    """
    nn = build_shallow_nn(X, y, num_hidden)
    with nn:
        if vi:  # variational inference (fast)
            # common schedule for `scale` is 1 at the beginning and 0 at the end
            scale = theano.shared(pm.floatX(1))
            vi = pm.ADVI(cost_part_grad_scale=scale)
            pm.fit(n=num_scale1_iters, method=vi)
            scale.set_value(0)
            approx = pm.fit(n=num_scale0_iters)
            trace = approx.sample(draws=num_samples)
        else:   # NUTS (very slow)
            trace = pm.sample(num_samples)
    return format_trace(trace)


def build_shallow_nn(X, y, num_hidden=100):
    """
    Build basic shallow Bayesian neural network
    """
    num_features = X.shape[1]
    num_classes = len(np.unique(y))
    floatX = theano.config.floatX
    Xt = theano.shared(X)
    W1_init = np.random.randn(num_features, num_hidden).astype(floatX)
    W2_init = np.random.randn(num_hidden, num_classes).astype(floatX)
    with pm.Model() as model_nn:
        # priors
        W1 = pm.Normal('W1', 0, sd=100, shape=W1_init.shape, testval=W1_init)
        b1 = pm.Flat('b1', shape=num_hidden)
        W2 = pm.Normal('W2', 0, sd=100, shape=W2_init.shape, testval=W2_init)
        b2 = pm.Flat('b2', shape=num_classes)

        # deterministic transformations
        z1 = Xt.dot(W1) + b1
        a1 = pm.math.tanh(z1)
        z2 = a1.dot(W2) + b2
        p = tt.nnet.softmax(z2)
        
        # likelihood
        observed = pm.Categorical('obs', p=p, observed=y)
    return model_nn
