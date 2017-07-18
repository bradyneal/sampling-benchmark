"""
This module provides functions for easy separation of data based on what kind
of data it is. For example, it is helpful to distinguish between categorical
and non-categorical data as categorical data is often more appropriately
represented in one-hot encodings.
"""

import numpy as np


def logical_filter(l, logicals):
    """
    Filter the list l down to the elements that correspond to the True
    elements in the list logicals.
    """
    return [val for (val, b) in zip(l, logicals) if b]


def separate_categorical(X, categorical, column_names=None):
    """
    Separate the categorical variables from the non-categorical variables
    as specificed by the boolean 'categorical' list
    
    Args:
        X: data matrix
        categorical: list of booleans indicating which features are categorical
        column_names: names of features
        
    Returns:
        (categorical portion of X, non-categorical portion of X)
        
        OR, if columns_names is specified:
        ((categorical portion of X, names of categorical features),
         (non-categorical portion of X, names of non-categorical features))
        
    """
    non_categorical = np.logical_not(categorical)
    
    if column_names:
        names_categ = logical_filter(column_names, categorical)
        names_non_categ = logical_filter(column_names, non_categorical)
        return ((X[:, categorical], names_categ),
                (X[:, non_categorical], names_non_categ))
    else:
        return X[:, categorical], X[:, non_categorical]


def separate_discrete(X, column_names=None):
    """
    Separate the discrete variables from the continuous variables (assuming all
    discrete variables are integers and everything else is a continuous
    variable)
    
    Args:
        X: data matrix
        column_names: names of features
        
    Returns:
        ("discrete" portion of X, "continuous" portion of X)
        
        OR, if columns_names is specified:
        (("discrete" portion of X, corresponding feature names),
         ("continuous" portion of X, corresponding feature names))
    """
    is_integer = np.vectorize(lambda x: x.is_integer(), otypes=[np.bool])

    def is_int_column(col):
        return np.all(is_integer(col))
    
    discrete = np.apply_along_axis(is_int_column, 0, X)
    continuous = np.logical_not(discrete)
    
    if column_names:
        names_discrete = logical_filter(column_names, discrete)
        names_continuous = logical_filter(column_names, continuous)
        return ((X[:, discrete], names_discrete),
                (X[:, continuous], names_continuous))
    else:
        return X[:, discrete], X[:, continuous]


def separate_features(X, categorical, column_names=None):
    """
    Separate data into categorical, discrete, and continuous variables
    (with the same assumptions as in separate_discrete())
    
    Args:
        X: data matrix
        categorical: list of booleans indicating which features are categorical
        column_names: names of features
        
    Returns:
        (categorical portion of X,
         "discrete" portion of X,
         "continuous" portion of X)
         
         OR, if columns_names is specified:
         ((categorical portion of X, corresponding feature names)
          ("discrete" portion of X, corresponding feature names)
          ("continuous" portion of X, corresponding feature names))
    """
    if column_names:
        (X_categ, names_categ), (X_non_categ, names_non_categ) = \
            separate_categorical(X, categorical, column_names)
        (X_discrete, names_discrete), (X_continuous, names_continuous) = \
            separate_discrete(X_non_categ, names_non_categ)
        return ((X_categ, names_categ),
                (X_discrete, names_discrete),
                (X_continuous, names_continuous))
    else:
        X_categ, X_non_categ = separate_categorical(X, categorical)
        X_discrete, X_continuous = separate_discrete(X_non_categ)
        return X_categ, X_discrete, X_continuous
    