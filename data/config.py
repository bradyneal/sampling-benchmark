"""
File for data config
"""

import os
from enum import Enum

UNIX_OPENML_PATH = '/data/lisa/data/openml'
OPENML_FOLDER = os.path.join(os.sep, *UNIX_OPENML_PATH.split('/'))
DATASETS_FOLDER = os.path.join(OPENML_FOLDER, 'datasets')
ERRORS_FOLDER = os.path.join(DATASETS_FOLDER, 'errors')

# Folder name constants
class Preprocess(Enum):
    RAW = 'raw'
    ONEHOT = 'one-hot'
    STANDARDIZED = 'standardized'
    ROBUST = 'robust_standardized'
    WHITENED = 'whitened'


CONFIG = {
    preprocess.value + '_folder': os.path.join(DATASETS_FOLDER, preprocess.value)
    for preprocess in Preprocess
}
CONFIG['errors_folder'] = ERRORS_FOLDER
