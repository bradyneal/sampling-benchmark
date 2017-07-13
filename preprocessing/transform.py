"""

"""

from sklearn.preprocessing import OneHotEncoder


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
