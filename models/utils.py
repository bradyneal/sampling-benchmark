"""
This module provides functions that help with construction of the models
in this package.
"""

from pymc3.backends.tracetab import trace_to_dataframe
from itertools import combinations
import pandas as pd


# NOTE: Current converts to a DataFrame and then to a numpy array.
# Consider doing it all in one go.
def format_trace(trace):
    """
    Convert the trace into the necessary format. The current format is a
    numpy array.
    """
    return pd.DataFrame.as_matrix(trace_to_dataframe(trace))


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


def make_col_names(d):
    """Make column names (i.e. x1, x2, ..., xd) for data dimension d"""
    return ['x' + str(i) for i in range(1, d + 1)]
