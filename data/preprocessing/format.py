"""
This module provides functions for changing between different data formats
(e.g. numpy ndarray and pandas DataFrame).
"""

import numpy as np
import pandas as pd
import scipy.sparse as sps

def numpy_to_dataframe(X, y=None):
    """Convert numpy 2d array to pandas DataFrame"""
    if y is None:
        data = X
        names = get_var_names(X.shape[1])
    else:
        data = np.concatenate((X, y[:, None]), axis=1)
        names = get_var_names(X.shape[1]) + ['y']
    df = pd.DataFrame(data=data, columns=names)
    return df


def to_ndarray(X):
    """
    Convert to numpy ndarray if not already. Right now, this only converts
    from sparse arrays.
    """
    # TODO need bool return?? rename as_full().
    if isinstance(X, np.ndarray):
        return X, False
    elif sps.issparse(X):
        print('Converting from sparse type: {}'.format(type(X)))
        return X.toarray(), True
    else:
        raise ValueError('Unexpected data type: {}'.format(type(X)))
    
    
def to_sparse(X):
    """Convert numpy ndarray to sparse matrix"""
    # TODO why does this need to be subroutine??
    # or just to_sparse = sps.csr_matrix
    return sps.csr_matrix(X)

def get_var_names(d):
    """Make column names (i.e. x1, x2, ..., xd) for data dimension d"""
    # TODO rename make col names, could also do based on passing in X
    # also zfill with 0 for sorting purposes
    # use '%0.nd' % d
    # why not start at 0??
    return ['x' + str(i) for i in range(1, d + 1)]
