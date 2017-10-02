"""
This module provides the sampling, given a model and step.
"""

import pymc3 as pm
from timeit import default_timer as timer

from .utils import format_trace
from . import MAX_NUM_SAMPLES, NUM_INIT_STEPS, SOFT_MAX_TIME_IN_SECONDS, \
              HARD_MAX_TIME_IN_SECONDS, MIN_SAMPLES_CONSTANT, NUM_CHAINS


def sample_model(model, step=None):
    """Sample from constructed Bayesian model"""
    start = timer()
    with model:
        pm._log.info('Auto-assigning NUTS sampler...')
        if step is None:
            start_, step = pm.init_nuts(init='advi', njobs=1, n_init=NUM_INIT_STEPS,
                                        random_seed=-1, progressbar=False)
        
        for i, trace in enumerate(pm.iter_sample(MAX_NUM_SAMPLES, step)):
            if i == 0:
                min_num_samples = MIN_SAMPLES_CONSTANT * (len(trace[0]) ** 2)
            elapsed = timer() - start
            if elapsed > SOFT_MAX_TIME_IN_SECONDS:
                print('exceeded soft time limit...')
                if i + 1 >= min_num_samples:
                    print('collected enough samples; stopping')
                    break
                else:
                    print('but only collected {} of {}; continuing...'
                          .format(i + 1, min_num_samples))
                    if elapsed > HARD_MAX_TIME_IN_SECONDS:
                        print('exceeded HARD time limit; STOPPING')
                        return None
    return format_trace(trace)
            