"""
This module provides functions for sampling from the posteriors of various
unsupervised models. The supported models are specified in the
UNSUPERVISED_MODEL_NAMES constant.
"""

import pymc3 as pm

UNSUPERVISED_MODEL_NAMES = []
