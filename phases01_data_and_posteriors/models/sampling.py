"""
This module provides the sampling, given a model and step.
"""

import pymc3 as pm
import theano
from timeit import default_timer as timer

from .utils import format_trace
from . import MAX_NUM_SAMPLES, NUM_INIT_STEPS, SOFT_MAX_TIME_IN_SECONDS, \
              HARD_MAX_TIME_IN_SECONDS, MIN_SAMPLES_CONSTANT, NUM_CHAINS, \
              NUM_SCALE1_ITERS, NUM_SCALE0_ITERS


def sample_model(model, step=None, num_samples=MAX_NUM_SAMPLES, advi=False,
                 num_scale1_iters=NUM_SCALE1_ITERS,
                 num_scale0_iters=NUM_SCALE0_ITERS):
    """Sample from constructed Bayesian model"""
    start = timer()
    with model:
        if not advi:
            pm._log.info('Assigning NUTS sampler...')
            if step is None:
                start_, step = pm.init_nuts(init='advi', njobs=1, n_init=NUM_INIT_STEPS,
                                            random_seed=-1, progressbar=False)
            
            for i, trace in enumerate(pm.iter_sample(num_samples, step)):
                if i == 0:
                    min_num_samples = get_min_num_samples(len(trace[0]),
                                                          MIN_SAMPLES_CONSTANT)
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
        else:   # ADVI for neural networks
            scale = theano.shared(pm.floatX(1))
            vi = pm.ADVI(cost_part_grad_scale=scale)
            pm.fit(n=num_scale1_iters, method=vi)
            scale.set_value(0)
            approx = pm.fit(n=num_scale0_iters)
            # one sample to get dimensions of trace
            trace = approx.sample(draws=1)
            min_num_samples = get_min_num_samples(len(trace.varnames),
                                                  MIN_SAMPLES_CONSTANT)
            trace = approx.sample(draws=min_num_samples)
    return format_trace(trace)


def get_min_num_samples(dimension, min_samples_constant):
    return min_samples_constant * (dimension ** 2)
