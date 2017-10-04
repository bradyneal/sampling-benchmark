"""
This module provides the sampling, given a model and step.
"""

import pymc3 as pm
from pymc3.backends.base import merge_traces
import theano
from timeit import default_timer as timer
from functools import partial
import pandas as pd

from .utils import format_trace
from . import MAX_NUM_SAMPLES, NUM_INIT_STEPS, SOFT_MAX_TIME_IN_SECONDS, \
              HARD_MAX_TIME_IN_SECONDS, MIN_SAMPLES_CONSTANT, NUM_CHAINS, \
              NUM_SCALE1_ITERS, NUM_SCALE0_ITERS


def sample_model(model, step=None, num_samples=MAX_NUM_SAMPLES, advi=False,
                 n_chains=NUM_CHAINS, num_scale1_iters=NUM_SCALE1_ITERS,
                 num_scale0_iters=NUM_SCALE0_ITERS):
    """Sample parallel chains from constructed Bayesian model"""
    sample_chain_with_args = partial(
        sample_chain, step=step, num_samples=num_samples, advi=advi,
        num_scale1_iters=num_scale1_iters, num_scale0_iters=num_scale0_iters)
  
    if advi:
        return format_trace(sample_chain_with_args(model), to_df=True)
    else:
        traces = []
        for i in range(n_chains):
            traces.append(sample_chain_with_args(model, i))
        multitrace = merge_traces(traces)
        df = format_trace(multitrace, to_df=True)
        return augment_with_diagnostics(df, get_diagnostics(multitrace))


def sample_chain(model, chain_i=0, step=None, num_samples=MAX_NUM_SAMPLES,
                 advi=False, num_scale1_iters=NUM_SCALE1_ITERS,
                 num_scale0_iters=NUM_SCALE0_ITERS):
    """Sample single chain from constructed Bayesian model"""
    start = timer()
    with model:
        if not advi:
            pm._log.info('Assigning NUTS sampler...')
            if step is None:
                start_, step = pm.init_nuts(init='advi', njobs=1, n_init=NUM_INIT_STEPS,
                                            random_seed=-1, progressbar=False)
            
            for i, trace in enumerate(pm.iter_sample(
                num_samples, step, chain=chain_i)):
                if i == 0:
                    min_num_samples = get_min_samples_per_chain(
                        len(trace[0]), MIN_SAMPLES_CONSTANT, NUM_CHAINS)
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
            min_num_samples = get_min_samples_per_chain(
                len(trace.varnames), MIN_SAMPLES_CONSTANT, 1)
            trace = approx.sample(draws=min_num_samples)
            
    return trace


def get_min_samples_per_chain(dimension, min_samples_constant, n_chains):
    return int(min_samples_constant * (dimension ** 2) / n_chains)


def augment_with_diagnostics(trace_df, diagnostics):
    """Add diagnostics to trace DataFrame"""
    d1, d2 = diagnostics
    if d1.keys() != d2.keys():
        raise ValueError('Diagnositics keys are not the same {} != {}'
                         .format(d1.keys(), d2.keys()))
    d1['diagnostic'] = 'Gelman-Rubin'
    d2['diagnostic'] = 'ESS'
    d_concat = {k: [d1[k], d2[k]] for k in d1.keys()}
    diag_df = pd.DataFrame.from_dict(d_concat)
    diag_df = df.set_index('diagnostic')
    df_concat = pd.concat([diag_df, trace_df])
    return df_concat


def get_diagnostics(trace):
    return pm.diagnostics.gelman_rubin(trace), pm.diagnostics.effective_n(trace)
