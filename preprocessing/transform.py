"""
This module provides functions for transforming data into a more useful
representation. For example, it does one-hot encoding of categorical variables
and standardization, or even whitening, of non-categorical variables.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .separation import separate_categorical


def one_hot(X, categorical=None):
    """
    Encode categorical variables in one-hot representation. If which variables
    are categorical is not specified, assume all variables are categorical.
    
    Args:
        X: data matrix
        categorical: list of booleans indicating which features are categorical
        
    Returns:
        one-hot representation of X
    """
    if categorical is None:
        categorical = [True] * X.shape[1]
    sparse_one_hot = OneHotEncoder(categorical_features=categorical) \
                        .fit_transform(X)
    return sparse_one_hot.toarray()


def standardize(X, scaler=None, return_scaler=False):
    """
    Standardize data (from each feature column, substract the corresponding
    mean and divide by the corresponding standard deviation). Assume it is
    appropriate to standardize every feature in X (e.g. X has no categorical
    variables).
    
    Args:
        X: data matrix
        scaler: if provided, use for standardization (e.g. for test set)
        return_scaler: if true, return scaler (to later be used on test set)
        
    Returns:
        standardized X and, optionally, the scaler object that contains the
        standardization information of X (training set)
    """
    if scaler is None:
        scaler = StandardScaler().fit(X)
    if return_scaler:
        return scaler.transform(X), scaler
    else:
        return scaler.transform(X)
    

def standardize_and_one_hot(X, categorical, scaler=None, return_scaler=False):
    """
    Standardize the non-categorical data and encode the categorical data in a
    one-hot representation. Note: this functions changes the order of the
    columns so that all of the non-categorical features come before the
    newly one-hot encoded categorical features.
    
    Args:
        X: data matrix
        categorical: list of booleans indicating which features are categorical
        scaler: if provided, use for standardization (e.g. for test set)
        return_scaler: if true, return scaler (to later be used on test set)
        
    Returns:
        standardized and one-hot encoded X and, optionally, the scaler object
        that contains the standardization information of X (training set)
    """
    X_categ, X_non_categ = separate_categorical(X, categorical)
    
    if return_scaler:
        X_scaled, scaler = standardize(X_non_categ, scaler, return_scaler)
    else:
        X_scaled = standardize(X_non_categ, scaler, return_scaler)
        
    X_one_hot = one_hot(X_categ)
    
    X_new = np.concatenate((X_scaled, X_one_hot), axis=1)
    return (X_new, scaler) if return_scaler else X_new
    