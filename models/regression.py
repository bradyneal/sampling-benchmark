"""
This module provides functions for sampling from the posteriors of various
regression models. The supported models are specified in the
REGRESSION_MODEL_NAMES constant.
"""

import pymc3 as pm
from pymc3.backends.tracetab import trace_to_dataframe
from itertools import combinations

from data import numpy_to_dataframe, get_var_names

REGRESSION_MODEL_NAMES = [
    'ls_linear', 'ls_pairwise_linear', 'ls_quadratic_linear',
    'robust_linear', 'robust_pairwise_linear', 'robust_quadratic_linear'
]
NUM_SAMPLES = 500


def sample_regression_model(model_name, X, y, num_samples=NUM_SAMPLES,
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
    if 'ls_linear' == model_name:
        sample_ls_linear(X, y, num_samples)
    elif 'ls_pairwise_linear' == model_name:
        sample_ls_pairwise(X, y, num_non_categorical, num_samples)
    elif 'ls_quadratic_linear' == model_name:
        sample_ls_quadratic(X, y, num_non_categorical, num_samples)
    elif 'robust_linear' == model_name:
        sample_robust_linear(X, y, num_samples)
    elif 'robust_pairwise_linear' == model_name:
        sample_robust_pairwise(X, y, num_non_categorical, num_samples)
    elif 'robust_quadratic_linear' == model_name:
        sample_robust_quadratic(X, y, num_non_categorical, num_samples)
    elif 'gp' == model_name:
        sample_gp(X, y, num_samples)
    else:
        raise ValueError('Unsupported model: {}\nSupported models: {}'
                         .format(model_name, MODEL_NAMES))

    
# GLM Defaults
# intercept:  flat prior
# weights:    N(0, 10^6) prior
# likelihood: Normal

def sample_linear(X, y, num_samples=NUM_SAMPLES, robust=False):
    """
    Sample from Bayesian linear model (abstraction of least squares and robust
    linear models that correspond to Normal and Student's T likelihoods
    respectively).
    """
    with pm.Model() as model_glm:
        if robust:
            pm.GLM(X, y, family=pm.glm.families.StudentT())
        else:
            pm.GLM(X, y)
        trace = pm.sample(num_samples)
    return format_trace(trace)


def sample_ls_linear(X, y, num_samples=NUM_SAMPLES):
    """
    Sample from Bayesian Least Squares Linear Regression. Uses Normal
    likelihood, which is equivalent to minimizing the mean squared error
    in the frequentist version of Least Squares.
    """
    return sample_linear(X, y, num_samples=NUM_SAMPLES)


def sample_robust_linear(X, y, num_samples=NUM_SAMPLES):
    """
    Sample from Bayesian Robust Regression. Uses Student's T likelihood as it
    has heavier tails than the Normal likelihood, allowing it to place less
    emphasis on outliers.
    """
    return sample_linear(X, y, num_samples=NUM_SAMPLES, robust=True)


def sample_interaction_linear(X, y, num_non_categorical=None,
                              num_samples=NUM_SAMPLES,
                              robust=False, interaction='pairwise'):
    """
    Sample from Bayesian linear models with interaction and higher order terms
    (abstraction of all linear regressions that have nonlinear data terms).
    
    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    d = X.shape[1]
    if num_non_categorical is None:
        num_non_categorical = d
    data_df = numpy_to_dataframe(X, y)
    with pm.Model() as model_glm:
        if 'pairwise' == interaction:
            interaction_formula = get_pairwise_formula(num_non_categorical)
        elif 'quadratic' == interaction:
            interaction_formula = get_quadratic_formula(num_non_categorical)
        x_formula = _join_nonempty(
            (interaction_formula,
             get_linear_formula(num_non_categorical + 1, d))
        )
        if robust:
            pm.GLM.from_formula('y ~ ' + x_formula, data_df,
                                family=pm.glm.families.StudentT())
        else:
            pm.GLM.from_formula('y ~ ' + x_formula, data_df)
        trace = pm.sample(num_samples)
    return format_trace(trace)
    

def sample_ls_pairwise(X, y, num_non_categorical=None,
                       num_samples=NUM_SAMPLES):
    """
    Sample from Bayesian Least Squares Linear Regression that has pairwise
    interaction terms between all non-categorial variables. Uses Normal
    likelihood, which is equivalent to minimizing the mean squared error
    in the frequentist version of Least Squares.
    
    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    return sample_interaction_linear(
        X, y, num_non_categorical=num_non_categorical,
        num_samples=num_samples, interaction='pairwise')
    

def sample_ls_quadratic(X, y, num_non_categorical=None,
                        num_samples=NUM_SAMPLES):
    """
    Sample from Bayesian Least Squares Linear Regression that has all first and
    second order non-categorical data terms. Uses Normal likelihood, which is
    equivalent to minimizing the mean squared error in the frequentist version
    of Least Squares.
    
    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    return sample_interaction_linear(
        X, y, num_non_categorical=num_non_categorical,
        num_samples=num_samples, interaction='quadratic')


def sample_robust_pairwise(X, y, num_non_categorical=None,
                           num_samples=NUM_SAMPLES):
    """
    Sample from Bayesian Robust Linear Regression that has pairwise
    interaction terms between all non-categorial variables. Uses Student's T
    likelihood as it has heavier tails than the Normal likelihood, allowing it
    to place less emphasis on outliers.
    
    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    return sample_interaction_linear(
        X, y, num_non_categorical=num_non_categorical, robust=True,
        num_samples=num_samples, interaction='pairwise')
    

def sample_robust_quadratic(X, y, num_non_categorical=None,
                            num_samples=NUM_SAMPLES):
    """
    Sample from Bayesian Robust Linear Regression that has all first and
    second order non-categorical data terms. Uses Student's T likelihood as it
    has heavier tails than the Normal likelihood, allowing it to place less
    emphasis on outliers.
    
    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    return sample_interaction_linear(
        X, y, num_non_categorical=num_non_categorical, robust=True,
        num_samples=num_samples, interaction='quadratic')


def get_pairwise_formula(num_non_categorical):
    var_names = get_var_names(num_non_categorical)
    singles_str = ' + '.join(var_names)
    pairs_str = ' + '.join(':'.join(pair)
                           for pair in combinations(var_names, 2))
    return _join_nonempty((singles_str, pairs_str))


def get_quadratic_formula(num_non_categorical):
    pairwise_str = get_pairwise_formula(num_non_categorical)
    var_names = get_var_names(num_non_categorical)
    squares_str = ' + '.join('np.power(x{}, 2)'.format(i)
                             for i in range(1, num_non_categorical + 1))
    return _join_nonempty((pairwise_str, squares_str))


def get_linear_formula(start_i, end_i):
    return ' + '.join('x' + str(i) for i in range(start_i, end_i + 1))


def _join_nonempty(l):
    return ' + '.join(s for s in l if s != '')


def sample_gp(X, y, num_samples=NUM_SAMPLES):
    """Sample from Gaussian Process"""
    raise NotImplementedError(
        'The Gaussian Process model is not yet implemented.')


def format_trace(trace):
    """
    Convert the trace into the necessary format. The current format is a
    Pandas DataFrame.
    """
    return trace_to_dataframe(trace)
    