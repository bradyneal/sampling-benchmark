"""
This module provides functions for sampling from the posteriors of various
regression models. The supported models are specified in the
REGRESSION_MODEL_NAMES constant.
"""

import pymc3 as pm
import theano.tensor as tt
from timeit import default_timer as timer

from data.preprocessing.format import numpy_to_dataframe
from .utils import format_trace, get_pairwise_formula, get_quadratic_formula, \
                   get_linear_formula, join_nonempty
from .nn import sample_shallow_nn
from . import MAX_NUM_SAMPLES, MAX_TIME_IN_SECONDS
from .utils import reduce_data_dimension

REGRESSION_MODEL_NAMES = \
    [
     'ls-linear-regres', 'robust-linear-regres', 'ls-quadratic-linear-regres',
     # 'ls-pairwise-linear-regres', 'robust-pairwise-linear-regres', 'robust-quadratic-linear-regres',
     # 'shallow-nn-regres',
     # 'gp-ExpQuad-regres', 'gp-Exponential-regres', 'gp-Matern32-regres','gp-Matern52-regres',
     # 'gp-RatQuad-regres' # currently throws an error
     ]


def sample_regression_model(model_name, X, y, num_samples=MAX_NUM_SAMPLES,
                            time=MAX_TIME_IN_SECONDS, step=None,
                            num_non_categorical=None):
    """
    Sample from the posteriors of any of the supported models

    Args:
        model_name: to specify which model to sample from
        X: data matrix
        y: targets
        num_samples: number points to sample from the model posterior
        step: type of PyMC3 sampler to use
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
    
    model_name = model_name.replace('_regres', '')
    
    # Build model
    if 'ls_linear' == model_name:
        model = build_ls_linear(X, y)
    elif 'ls_pairwise_linear' == model_name:
        model = build_ls_pairwise(X, y, num_non_categorical)
    elif 'ls_quadratic_linear' == model_name:
        model = build_ls_quadratic(X, y, num_non_categorical)
    elif 'robust_linear' == model_name:
        model = build_robust_linear(X, y)
    elif 'robust_pairwise_linear' == model_name:
        model = build_robust_pairwise(X, y, num_non_categorical)
    elif 'robust_quadratic_linear' == model_name:
        model = build_robust_quadratic(X, y, num_non_categorical)
    elif 'shallow_nn' == model_name:
        return sample_shallow_nn_regres(X, y, num_samples)
    elif 'gp_ExpQuad' == model_name:
        model = build_gp(X, y, 'ExpQuad')
    elif 'gp_Exponential' == model_name:
        model = build_gp(X, y, 'Exponential')
    elif 'gp_Matern32' == model_name:
        model = build_gp(X, y, 'Matern32')
    elif 'gp_Matern52' == model_name:
        model = build_gp(X, y, 'Matern52')
    elif 'gp_RatQuad' == model_name:
        model = build_gp(X, y, 'RatQuad')
    else:
        raise ValueError('Unsupported model: {}\nSupported models: {}'
                         .format(model_name, REGRESSION_MODEL_NAMES))
    
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
                print('exceeded max time... stopping')
                break
    return format_trace(trace)
            


# GLM Defaults
# intercept:  flat prior
# weights:    N(0, 10^6) prior
# likelihood: Normal

def build_linear(X, y, robust=False):
    """
    Build Bayesian linear model (abstraction of least squares and robust
    linear models that correspond to Normal and Student's T likelihoods
    respectively).
    """
    with pm.Model() as model_linear:
        if robust:
            pm.glm.GLM(X, y, family=pm.glm.families.StudentT())
        else:
            pm.glm.GLM(X, y)
    return model_linear


def build_ls_linear(X, y):
    """
    Build Bayesian Least Squares Linear Regression. Uses Normal
    likelihood, which is equivalent to minimizing the mean squared error
    in the frequentist version of Least Squares.
    """
    return build_linear(X, y)


def build_robust_linear(X, y):
    """
    Build Bayesian Robust Regression. Uses Student's T likelihood as it
    has heavier tails than the Normal likelihood, allowing it to place less
    emphasis on outliers.
    """
    return build_linear(X, y, robust=True)


def build_interaction_linear(X, y, num_non_categorical=None,
                             robust=False, interaction='pairwise'):
    """
    Build Bayesian linear models with interaction and higher order terms
    (abstraction of all linear regressions that have nonlinear data terms).

    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    d = X.shape[1]
    if num_non_categorical is None:
        num_non_categorical = d
    data_df = numpy_to_dataframe(X, y)
    with pm.Model() as model_interaction:
        if 'pairwise' == interaction:
            interaction_formula = get_pairwise_formula(num_non_categorical)
        elif 'quadratic' == interaction:
            interaction_formula = get_quadratic_formula(num_non_categorical)
        x_formula = join_nonempty(
            (interaction_formula,
             get_linear_formula(num_non_categorical + 1, d))
        )
        if robust:
            pm.glm.GLM.from_formula('y ~ ' + x_formula, data_df,
                                family=pm.glm.families.StudentT())
        else:
            pm.glm.GLM.from_formula('y ~ ' + x_formula, data_df)
    return model_interaction


def build_ls_pairwise(X, y, num_non_categorical=None):
    """
    Build Bayesian Least Squares Linear Regression that has pairwise
    interaction terms between all non-categorial variables. Uses Normal
    likelihood, which is equivalent to minimizing the mean squared error
    in the frequentist version of Least Squares.

    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    return build_interaction_linear(
        X, y, num_non_categorical=num_non_categorical, interaction='pairwise')


def build_ls_quadratic(X, y, num_non_categorical=None):
    """
    Build Bayesian Least Squares Linear Regression that has all first and
    second order non-categorical data terms. Uses Normal likelihood, which is
    equivalent to minimizing the mean squared error in the frequentist version
    of Least Squares.

    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    return build_interaction_linear(
        X, y, num_non_categorical=num_non_categorical, interaction='quadratic')


def build_robust_pairwise(X, y, num_non_categorical=None):
    """
    Build Bayesian Robust Linear Regression that has pairwise
    interaction terms between all non-categorial variables. Uses Student's T
    likelihood as it has heavier tails than the Normal likelihood, allowing it
    to place less emphasis on outliers.

    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    return build_interaction_linear(
        X, y, num_non_categorical=num_non_categorical, robust=True,
        interaction='pairwise')


def build_robust_quadratic(X, y, num_non_categorical=None):
    """
    Build Bayesian Robust Linear Regression that has all first and
    second order non-categorical data terms. Uses Student's T likelihood as it
    has heavier tails than the Normal likelihood, allowing it to place less
    emphasis on outliers.

    Assumptions:
    1. all non-categorial variables come first in the ordering
    2. all variables are non-categorical if num_non_categorical isn't given
    """
    return build_interaction_linear(
        X, y, num_non_categorical=num_non_categorical, robust=True,
        interaction='quadratic')


def sample_shallow_nn_regres(X, y, num_samples=MAX_NUM_SAMPLES):
    """
    Sample from shallow Bayesian neural network, using variational inference.
    Uses Normal likelihood.
    """
    return sample_shallow_nn(X, y, 'regression')


def build_pm_gp_cov(D, cov_f='ExpQuad'):
    cov_dict = {'ExpQuad': pm.gp.cov.ExpQuad,
                'Exponential': pm.gp.cov.Exponential,
                'Matern32': pm.gp.cov.Matern32,
                'Matern52': pm.gp.cov.Matern52}

    log_ls = pm.Uniform('log_ls', lower=-2.0, upper=3.0)
    ls = pm.Deterministic('ls', tt.exp(log_ls))

    if cov_f == 'RatQuad':
        log_alpha = pm.Uniform('log_alpha', lower=-2.0, upper=5.0)
        alpha = pm.Deterministic('alpha', tt.exp(log_alpha))
        K = pm.gp.cov.RatQuad(D, ls=ls, alpha=alpha)
    else:
        assert(cov_f in cov_dict)  # TODO change to some type of better error
        cov_class = cov_dict[cov_f]
        K = cov_class(D, ls)
    return K


def build_gp(X, y, cov_f='ExpQuad'):
    """Build Gaussian Process"""
    # TODO also implement version that uses Elliptical slice sampling
    N, D = X.shape

    with pm.Model() as model_gp:
        # uninformative prior on the function variance
        log_s2_f = pm.Uniform('log_s2_f', lower=-10.0, upper=5.0)
        s2_f = pm.Deterministic('s2_f', tt.exp(log_s2_f))

        # covariance functions for the function f and the noise
        f_cov = s2_f * build_pm_gp_cov(D, cov_f)

        # uninformative prior on the noise variance
        log_s2_n = pm.Uniform('log_s2_n', lower=-10.0, upper=5.0)
        s2_n = pm.Deterministic('s2_n', tt.exp(log_s2_n))

        pm.gp.GP('y_obs', cov_func=f_cov, sigma=s2_n,
                 observed={'X': X, 'Y': y})
        
    return model_gp
