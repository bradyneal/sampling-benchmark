"""
File for data config
"""

import os
import getpass
from enum import Enum


def path_from_unix_path(unix_path):
    return os.path.join(os.sep, *unix_path.split('/'))


UNIX_OPENML_PATH = '/data/lisa/data/openml'
LISATMP_NUM = 3
USERNAME = getpass.getuser()
PROJECT_NAME = 'sampling-benchmark'
UNIX_EXP_PATH = '/data/lisatmp{}/{}/{}'.format(LISATMP_NUM, USERNAME,
                                               PROJECT_NAME)
OPENML_FOLDER = path_from_unix_path(UNIX_OPENML_PATH)
DATASETS_FOLDER = os.path.join(OPENML_FOLDER, 'datasets')
ERRORS_FOLDER = os.path.join(DATASETS_FOLDER, 'errors')
TASKS_FOLDER = os.path.join(OPENML_FOLDER, 'tasks')
EXP_FOLDER = path_from_unix_path(UNIX_EXP_PATH)
SAMPLES_FOLDER = os.path.join(EXP_FOLDER, 'samples')
DIAGNOSTICS = os.path.join(SAMPLES_FOLDER, 'diagnostics.pkl')

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
CONFIG['diagnostics'] = DIAGNOSTICS

# Prepare directories:
if not os.path.exists(SAMPLES_FOLDER):
    os.makedirs(SAMPLES_FOLDER)
