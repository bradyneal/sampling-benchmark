"""
This module provides functions for sampling from the posteriors of various
different models. The supported models are specified in the MODEL_NAMES
constant.
"""

import pymc3 as pm

MODEL_NAMES = ['ls_linear', 'robust_linear']
NUM_SAMPLES = 500


def sample_model(model_name, X, y, num_samples=NUM_SAMPLES):
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
    if 'ls_linear' == model_name:
        sample_ls_linear(X, y, num_samples)
    elif 'robust_linear' == model_name:
        sample_robust_linear(X, y, num_samples)
    elif 'gp' == model_name:
        sample_gp(X, y, num_samples)
    else:
        raise ValueError('Unsupported model: {}\nSupported models: {}'
                         .format(model_name, MODEL_NAMES))

    
# GLM Defaults
# intercept:  flat prior
# weights:    N(0, 10^6) prior
# likelihood: Normal

def sample_ls_linear(X, y, num_samples=NUM_SAMPLES):
    """
    Sample from Bayesian Least Squares Linear Regression. Uses Normal
    likelihood, which is equivalent to minimizing the mean squared error
    in the frequentist version of Least Squares.
    """
    with pm.Model() as model_glm:
        pm.GLM(X, y)
        trace = pm.sample(num_samples)
    return trace


def sample_robust_linear(X, y, num_samples=NUM_SAMPLES):
    """
    Sample from Bayesian Robust Regression. Uses Student's T likelihood as it
    has heavier tails than the Normal likelihood, allowing it to place less
    emphasis on outliers.
    """
    with pm.Model() as model_glm:
        StudentT = pm.glm.families.StudentT()
        pm.GLM(X, y, family=StudentT)
        trace = pm.sample(num_samples)
    return trace


def sample_gp(X, y, num_samples=NUM_SAMPLES):
    """Sample from Gaussian Process"""
    raise NotImplementedError(
        'The Gaussian Process model is not yet implemented.')
