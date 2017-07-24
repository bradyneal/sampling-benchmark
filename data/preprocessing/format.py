"""
This module provides functions for changing between different data formats
(e.g. numpy ndarray and pandas DataFrame).
"""

import numpy as np
import pandas as pd


def numpy_to_dataframe(X, y=None):
    """Convert numpy 2d array to pandas DataFrame"""
    if y is not None:
        y = np.reshape(y, (len(y), 1))
        data = np.concatenate((X, y), axis=1)
        names = get_var_names(X.shape[1]) + ['y']
    else:
        data = X
        names = get_var_names(X.shape[1])
    return pd.DataFrame(data, columns=names)


def get_var_names(d):
    """Make column names (i.e. x1, x2, ..., xd) for data dimension d"""
    return ['x' + str(i) for i in range(1, d + 1)]
