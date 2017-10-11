# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import os
import sys
import warnings
from emcee.autocorr import integrated_time
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.backends.tracetab import trace_to_dataframe
import theano
import theano.tensor as T
from models import BUILD_MODEL, SAMPLE_MODEL
from samplers import BUILD_STEP_PM, BUILD_STEP_MC
from chunker import time_chunker
from chunker import CHUNK_SIZE, GRID_INDEX
import fileio as io
# These modules should be replaced with better options if phase3 goes Python3
from time import time as wall_time
from time import clock as cpu_time

from sklearn.preprocessing import StandardScaler
import scipy.stats as ss

SAMPLE_INDEX_COL = 'sample'
DATA_CENTER = 'data_center'
DATA_SCALE = 'data_scale'
MAX_N = 10 ** 5  # Some value to prevent blowing out HDD space with samples.


def format_trace(trace):
    df = trace_to_dataframe(trace)
    return df.values


def moments_report(X, epsilon=1e-12):
    # TODO eliminate repetition with phase 2 here
    N, D = X.shape

    finite = np.all(np.isfinite(X))

    acc = np.abs(np.diff(X, axis=0)) > epsilon
    acc_valid = np.all(np.any(acc, 1) == np.all(acc, 1))
    acc_rate = np.mean(acc[:, 0])

    print 'N = %d, D = %d' % (N, D)
    print 'finite %d, accept %d' % (finite, acc_valid)
    print 'acc rate %f' % acc_rate

    V = np.std(X, axis=0)
    std_ratio = np.log10(np.max(V) / np.min(V))

    C = np.cov(X, rowvar=0)
    cond_number = np.log10(np.linalg.cond(C))

    corr = np.corrcoef(X, rowvar=0) - np.eye(X.shape[1])

    max_skew = np.max(np.abs(ss.skew(X, axis=0)))
    max_kurt = np.max(ss.kurtosis(X, axis=0))

    print 'log10 std ratio %f, cond number %f' % (std_ratio, cond_number)
    print 'min corr %f, max corr %f' % (np.min(corr), np.max(corr))
    print 'max skew %f, max kurt %f' % (max_skew, max_kurt)

# ============================================================================
# Part of Fred's funky system to keep track of Theano function evals which
# needs global variables. In the future this can be eliminated, we hope.

all_counters = []


def reset_counters():
    del all_counters[:]  # reset to empty list


def get_counters():
    L = len(all_counters)
    assert(L >= 1)
    if L > 1:
        warnings.warn('Usually there is only 1 counter, now there are %d' % L)
    count = sum(int(c.get_value()) for c in all_counters)
    return count

# ============================================================================


def sample_pymc3(logpdf_tt, sampler, start, timers, time_grid_ms, n_grid,
                 data_scale=None):
    assert(start.ndim == 1)
    D, = start.shape

    with pm.Model():
        pm.DensityDist('x', logpdf_tt, shape=D)

        step_kwds = {}
        if data_scale is not None:
            assert(data_scale.shape == (D,))
            step_kwds['scaling'] = data_scale
        print 'step arguments'
        print step_kwds

        steps = BUILD_STEP_PM[sampler](step_kwds)

        sample_gen = pm.sampling.iter_sample(MAX_N, steps, start={'x': start})

        time_grid_s = 1e-3 * time_grid_ms
        TC = time_chunker(sample_gen, time_grid_s, timers, n_grid=n_grid)

        print 'starting to sample'
        # This could all go in a list comp if we get rid of the assert check
        cum_size = 0
        meta = []
        for trace, metarow in TC:
            meta.append(metarow)
            cum_size += metarow[CHUNK_SIZE]
            assert(cum_size == len(trace) - 1)
    # Build rep for trace data
    trace = format_trace(trace)
    return trace, meta


def sample_emcee(logpdf_tt, sampler, start, timers, time_grid_ms, n_grid,
                 n_walkers_min=50, thin=100, data_scale=None, ball_size=1e-6):
    '''Use default thin of 100 since otherwise too fast and could blow out
    memory with samples on high time limit.'''
    assert(start.ndim == 1)
    D, = start.shape
    data_scale = np.ones(D) if data_scale is None else data_scale
    assert(data_scale.shape == (D,))

    n_walkers = max(2 * D + 2, n_walkers_min)
    ball = (ball_size * data_scale[None, :]) * np.random.randn(n_walkers, D)
    start = ball + start[None, :]

    # emcee does not need gradients so we could pass np only implemented
    # version if that is less overhead, but not that is not clear. So, just
    # compile the theano version.
    x_tt = T.vector('x')
    x_tt.tag.test_value = np.zeros(D)
    logpdf_val = logpdf_tt(x_tt)
    logpdf_f = theano.function([x_tt], logpdf_val)

    print 'running emcee with %d, %d' % (n_walkers, D)
    sampler_obj = BUILD_STEP_MC[sampler](n_walkers, D, logpdf_f)

    print 'doing init'
    # Might want to consider putting save chain to false since emcee uses
    # np.concat to grow chain. Might be less overhead to append to list in the
    # loop below.
    sample_gen = sampler_obj.sample(start, iterations=MAX_N * thin, thin=thin,
                                    storechain=True)

    time_grid_s = 1e-3 * time_grid_ms
    TC = time_chunker(sample_gen, time_grid_s, timers, n_grid=n_grid)

    print 'starting to sample'
    # This could all go in a list comp if we get rid of the assert check
    cum_size = 0
    meta = []
    for trace, metarow in TC:
        meta.append(metarow)
        cum_size += metarow[CHUNK_SIZE]
        assert(sampler_obj.chain.shape == (n_walkers, MAX_N, D))
    # Build rep for trace data
    # Same as:
    # np.concatenate([X[ii, :, :] for ii in xrange(X.shape[0])], axis=0)
    # EnsembleSampler.flatchain does this too but doesn't truncate at cum_size
    trace = np.reshape(sampler_obj.chain[:, :cum_size, :], (-1, D))
    # assert(trace.shape == (cum_size * n_walkers, D))

    # Log the emcee version of autocorr for future ref
    try:
        tau = integrated_time(trace, axis=0)
        print 'flat auto-corr'
        print tau
    except Exception as err:
        print 'emcee autocorr est failed'
        print str(err)

    return trace, meta


def init_setup(logpdf_tt, D, init='advi'):
    with pm.Model():
        pm.DensityDist('x', logpdf_tt, shape=D)
        start, step = pm.sampling.init_nuts(init, progressbar=False)
    start = start['x']
    scale = step.potential.s
    return start, scale


def controller(model_setup, sampler, time_grid_ms, n_grid,
               start_mode='default', scale_mode='default', n_ref_exact=1000):
    assert(time_grid_ms > 0)

    model_name, D, params_dict = model_setup
    assert(model_name in BUILD_MODEL)

    timers = [('chunk_cpu_time_s', cpu_time),
              ('chunk_wall_time_s', wall_time),
              ('energy_calls', get_counters)]

    print '-' * 20
    print 'starting experiment'
    print model_name
    print sampler
    print 'D=%d' % D
    assert(D >= 1)
    assert(params_dict[DATA_CENTER].shape == (D,))
    assert(params_dict[DATA_SCALE].shape == (D,))

    # Use default arg trick to get params to bind to model now
    def logpdf(x, p=params_dict):
        # This is Fred's trick to implicitly count function evals in theano.
        s = theano.shared(0, name='function_calls')
        all_counters.append(s)
        s.default_update = s + 1

        # Benchmark was trained on standardized data, but we want to sample in
        # scale of original problem to be realistic.
        x_std = (x - p[DATA_CENTER]) / p[DATA_SCALE]
        ll = BUILD_MODEL[model_name](x_std, p)
        # This constant offset actually cancels in MCMC, but might as well do
        # the logpdf correctly to avoid secretly leaking scale information. We
        # might want to consider adding random shifts since real densities are
        # not normalized.
        ll = ll - np.sum(np.log(p[DATA_SCALE]))
        return ll + s * 0

    # Process input options for initialization
    if start_mode == 'advi' or scale_mode == 'advi':
        advi_start, advi_scale = init_setup(logpdf, D)

    start = None
    if start_mode == 'exact':
        start = sample_exact(model_name, D, params_dict, N=1)[0, :]
        assert(start.shape == (D,))
    elif start_mode == 'advi':
        start = advi_start
        assert(start.shape == (D,))
    else:
        assert(start_mode == 'default')

    data_scale = None
    if scale_mode == 'exact':
        data_scale = params_dict[DATA_SCALE]
        assert(data_scale.shape == (D,))
    elif scale_mode == 'advi':
        data_scale = advi_scale
        assert(data_scale.shape == (D,))
    else:
        assert(scale_mode == 'default')

    reset_counters()
    if sampler in BUILD_STEP_PM:
        trace, meta = sample_pymc3(logpdf, sampler, start,
                                   timers, time_grid_ms, n_grid,
                                   data_scale)
    else:
        assert(sampler in BUILD_STEP_MC)
        # Intentionally not passing data_scale, since emcee doesn't seem to
        # have a good way to use it, built in.
        trace, meta = sample_emcee(logpdf, sampler, start,
                                   timers, time_grid_ms, n_grid)
    moments_report(trace)

    if n_ref_exact > 0:
        X_exact = sample_exact(model_name, D, params_dict, N=n_ref_exact)
        print 'std exact'
        print np.std(X_exact, axis=0)

        scaler = StandardScaler()
        X_exact = scaler.fit_transform(X_exact)
        X_std = scaler.transform(trace)
        err = np.mean((np.mean(X_exact, axis=0) - np.mean(X_std, axis=0)) ** 2)
        print 'sq err %f' % err

    # Build a meta-data df
    meta = pd.DataFrame(meta)
    meta.set_index(GRID_INDEX, drop=True, inplace=True)
    assert(meta.index[0] == 0 and meta.index[-1] < n_grid)
    assert(np.all(np.diff(meta.index.values) > 0))
    assert(np.all(meta.values >= 0))  # Will also catch nans
    meta = meta.reindex(index=xrange(n_grid), fill_value=0)
    meta[SAMPLE_INDEX_COL] = meta[CHUNK_SIZE].cumsum()
    # Could assert iter and index dtype is int here to be really safe
    return trace, meta


def sample_exact(model_name, D, params_dict, N=1):
    X = SAMPLE_MODEL[model_name](params_dict, N=N)
    assert(X.shape == (N, D))
    # Benchmark model trained on standardized data, move back to original.
    X = params_dict[DATA_SCALE][None, :] * X + \
        params_dict[DATA_CENTER][None, :]
    return X


def run_experiment(config, param_name, sampler):
    assert(sampler == config['exact_name'] or
           (sampler in BUILD_STEP_PM) or (sampler in BUILD_STEP_MC))

    model_file = param_name + config['pkl_ext']
    model_file = os.path.join(config['input_path'], model_file)
    print 'loading %s' % model_file
    assert(os.path.isabs(model_file))
    with open(model_file, 'rb') as f:
        model_setup = pkl.load(f)
    model_name, D, params_dict = model_setup
    assert(model_name in SAMPLE_MODEL)

    # Now sample
    meta = None
    if sampler == config['exact_name']:
        X = sample_exact(model_name, D, params_dict, N=config['n_exact'])
    else:
        X, meta = controller(model_setup, sampler,
                             config['t_grid_ms'], config['n_grid'],
                             config['start_mode'], config['scale_mode'])
    # Now save the data
    data_file = io.build_output_name(param_name, sampler)
    data_file = io.get_temp_filename(config['output_path'], data_file,
                                     config['csv_ext'])
    print 'saving samples to %s' % data_file
    np.savetxt(data_file, X, delimiter=',')

    # Save meta data
    if meta is not None:
        meta_file = data_file + config['meta_ext']
        print 'saving meta-data to %s' % meta_file
        assert(not os.path.isfile(meta_file))  # This could be warning
        assert(not meta.isnull().any().any())
        meta.to_csv(meta_file, header=True, index=False)


def main():
    assert(len(sys.argv) == 4)
    config_file = io.abspath2(sys.argv[1])
    param_name = sys.argv[2]
    sampler = sys.argv[3]
    assert(io.is_safe_name(param_name))

    config = io.load_config(config_file)

    run_experiment(config, param_name, sampler)
    print 'done'

if __name__ == '__main__':
    main()
