"""
File for data config
"""

import os
from enum import Enum

UNIX_OPENML_PATH = '/data/lisa/data/openml'
UNIX_SAMPLES_PATH = '' # TBD
OPENML_FOLDER = path_from_unix_path(UNIX_OPENML_PATH)
DATASETS_FOLDER = os.path.join(OPENML_FOLDER, 'datasets')
ERRORS_FOLDER = os.path.join(DATASETS_FOLDER, 'errors')
TASKS_FOLDER = os.path.join(OPENML_FOLDER, 'tasks')
SAMPLES_FOLDER = path_from_unix_path(UNIX_SAMPLES_PATH)

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
CONFIG['tasks_folder'] = TASKS_FOLDER
CONFIG['samples_folder'] = SAMPLES_FOLDER

def path_from_unix_path(unix_path):
    return os.path.join(os.sep, *unix_path.split('/'))
