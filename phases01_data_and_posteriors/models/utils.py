"""
This module provides functions that help with construction of the models
in this package.
"""

from pymc3.backends.tracetab import trace_to_dataframe
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from . import MAX_DATA_DIMENSION, MAX_GP_N, MAX_N
from data.preprocessing.format import make_col_names


# NOTE: Current converts to a DataFrame and then to a numpy array.
# Consider doing it all in one go.
def format_trace(trace, to_df=True):
    """
    Convert the trace into the necessary format. The current format is a
    numpy array.
    """
    df = trace_to_dataframe(trace)
    if to_df:
        return df
    else:
        return pd.DataFrame.as_matrix(df)


def subsample(X, y, model_name):
    """Subsample X"""
    n = X.shape[0]
    if 'gp' in model_name and n > MAX_GP_N:
        idx = np.random.randint(n, size=MAX_GP_N)
        return X[idx, :], y[idx]
    elif n > MAX_N:
        idx = np.random.randint(n, size=MAX_N)
        return X[idx, :], y[idx]
    else:
        return X, y


def reduce_data_dimension(X, model_name, transform=None,
                          return_transform=False):
    """
    Linearly project the data down to a small enough dimension for the model
    to run in a reasonable amount of time. If the data is already small enough
    in dimension, just return the unaltered data.
    
    Args:
        X: data matrix
        model_type: type of Bayesian model
        transform: if provided, use for dim reduction (e.g. for test set)
        return_transform: if true, return transform (to later be used on test set)
        
    Returns:
        dimensionality reduced X and, optionally, the transform object that
        contains the PCA information of X (training set)
    """
    max_dimension = get_max_dimension(model_name)
    if X.shape[1] > max_dimension:
        if transform is None:
            transform = PCA(n_components=max_dimension)
            X_reduced = transform.fit_transform(X)
        else:
            X_reduced = transform.transform(X)
        return (X_reduced, transform) if return_transform else X_reduced
    else:
        return (X, None) if return_transform else X
    
    
def get_max_dimension(model_name):
    """Get the max dimension for the specified model type"""
    is_quadratic = 'quadratic' in model_name
    is_pairwise = 'pairwise' in model_name
    is_linear = 'linear' in model_name
    is_nn = 'nn' in model_name
    is_gp = 'gp' in model_name
    if is_quadratic:
        max_dimension = MAX_DATA_DIMENSION['quadratic']
    elif is_pairwise:
        max_dimension = MAX_DATA_DIMENSION['pairwise']
    elif is_linear:
        max_dimension = MAX_DATA_DIMENSION['linear']
    elif is_nn or is_gp:
        # Should this be changed?
        max_dimension = float('inf')
    else:
        raise ValueError('Invalid model: ' + model_name)
    return max_dimension
    

def get_pairwise_formula(num_non_categorical):
    """
    Get Patsy formula string that has all first order and interaction terms
    between variables.
    
    >>> get_pairwise_formula(3)
    'x1 + x2 + x3 + x1:x2 + x1:x3 + x2:x3'
    """
    var_names = make_col_names(num_non_categorical)
    singles_str = ' + '.join(var_names)
    pairs_str = ' + '.join(':'.join(pair)
                           for pair in combinations(var_names, 2))
    return join_nonempty((singles_str, pairs_str))


def get_quadratic_formula(num_non_categorical):
    """
    Get Patsy formula string that has all first order and second order terms.
    
    >>> get_quadratic_formula(2)
    'x1 + x2 + x1:x2 + np.power(x1, 2) + np.power(x2, 2)'
    """
    pairwise_str = get_pairwise_formula(num_non_categorical)
    var_names = make_col_names(num_non_categorical)
    squares_str = ' + '.join('np.power(x{}, 2)'.format(i)
                             for i in range(1, num_non_categorical + 1))
    return join_nonempty((pairwise_str, squares_str))


def get_linear_formula(start_i, end_i):
    """
    Get Patsy formula string that has the first order terms for the variables
    that range from start_i to end_i (inclusive).
    
    >>> get_linear_formula(4, 9)
    'x4 + x5 + x6 + x7 + x8 + x9'
    """
    return ' + '.join('x' + str(i) for i in range(start_i, end_i + 1))


def join_nonempty(l):
    """
    Join all of the nonempty string with a plus sign.
    
    >>> join_nonempty(('x1 + x2 + x1:x2', 'x3 + x4'))
    'x1 + x2 + x1:x2 + x3 + x4'
    >>> join_nonempty(('abc', '', '123', ''))
    'abc + 123'
    """
    return ' + '.join(s for s in l if s != '')
