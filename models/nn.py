"""
This module provides functions for building and sampling from the posteriors
of Bayesian neural networks of varying widths, depths, and output types
(e.g. regression or classification).
"""

import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt

from .utils import format_trace

SUPPORTED_OUTPUTS = ['regression', 'classification']
NUM_SAMPLES = 500
NUM_HIDDEN = 100
NUM_SCALE1_ITERS = 20000
NUM_SCALE0_ITERS = 30000


def sample_shallow_nn(
    X, y, output, num_hidden=NUM_HIDDEN, num_samples=NUM_SAMPLES, vi=True,
    num_scale1_iters=NUM_SCALE1_ITERS, num_scale0_iters=NUM_SCALE0_ITERS):
    """
    Sample from shallow Bayesian neural network
    """
    nn = build_shallow_nn(X, y, output, num_hidden)
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


def build_shallow_nn(X, y, output='regression', num_hidden=NUM_HIDDEN):
    """
    Build basic shallow Bayesian neural network
    """
    if output not in SUPPORTED_OUTPUTS:
        raise ValueError(
            'Unsupported neural network output: {}\nSupported outputs: {}'
            .format(output, SUPPORTED_OUTPUTS))
    if 'regression' == output:
        num_output_units = 1
    elif 'classification' == output:
        num_output_units = len(np.unique(y))
        
    num_features = X.shape[1]
    floatX = theano.config.floatX
    Xt = theano.shared(X)
    W1_init = np.random.randn(num_features, num_hidden).astype(floatX)
    W2_init = np.random.randn(num_hidden, num_output_units).astype(floatX)
    with pm.Model() as model_nn:
        # priors
        W1 = pm.Normal('W1', 0, sd=100, shape=W1_init.shape, testval=W1_init)
        b1 = pm.Flat('b1', shape=W1_init.shape[1])
        W2 = pm.Normal('W2', 0, sd=100, shape=W2_init.shape, testval=W2_init)
        b2 = pm.Flat('b2', shape=W2_init.shape[1])

        # deterministic transformations
        z1 = Xt.dot(W1) + b1
        a1 = pm.math.tanh(z1)
        z2 = a1.dot(W2) + b2
        
        # format output and plug in data
        if 'regression' == output:
            observed = pm.Normal('obs', mu=z2, sd=1000, observed=y)
        elif 'classification' == output:
            p = tt.nnet.softmax(z2)
            observed = pm.Categorical('obs', p=p, observed=y)
        
    return model_nn
