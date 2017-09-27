# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import os
import sys
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.backends.tracetab import trace_to_dataframe
import theano
from models import BUILD_MODEL, SAMPLE_MODEL
from samplers import BUILD_STEP
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
MAX_N = 1000000  # Some value to prevent blowing out HDD space with samples.


def format_trace(trace):
    # TODO I don't think we need to import extra function to get df from trace
    df = trace_to_dataframe(trace)
    return df.values


def moments_report(X):
    V = np.std(X, axis=0)
    std_ratio = np.log10(np.max(V) / np.min(V))

    C = np.cov(X, rowvar=0)
    cond_number = np.log10(np.linalg.cond(C))

    corr = np.corrcoef(X, rowvar=0) - np.eye(X.shape[1])

    max_skew = np.max(np.abs(ss.skew(X, axis=0)))
    max_kurt = np.max(ss.kurtosis(X, axis=0))

    print 'N = %d' % X.shape[0]
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
    assert(len(all_counters) == 1)  # For now, it seems this is way it goes
    count = int(all_counters[0].get_value())
    return count

# ============================================================================


def controller(model_setup, sampler, time_grid_ms, n_grid):
    assert(sampler in BUILD_STEP)
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

    # TODO remove debug only
    #params_dict[DATA_SCALE] = np.ones_like(params_dict[DATA_SCALE])
    # params_dict[DATA_CENTER] = np.zeros_like(params_dict[DATA_CENTER])
    # params_dict[DATA_SCALE] = np.minimum(10.0 * np.min(params_dict[DATA_SCALE]), params_dict[DATA_SCALE])
    # params_dict[DATA_SCALE] = np.logspace(-3, 3, D)
    print params_dict[DATA_CENTER]
    print params_dict[DATA_SCALE]

    # TODO test only remove
    X_exact = sample_exact(model_name, D, params_dict, N=10000)
    print 'exact'
    moments_report(X_exact)

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

    # TODO insert validation test here

    reset_counters()
    with pm.Model():
        pm.DensityDist('x', logpdf, shape=D)
        steps = BUILD_STEP[sampler]()

        print 'doing init'
        init_trace = pm.sample(1, steps, init='advi', start={'x': params_dict[DATA_CENTER]})
        sample_generator = pm.sampling.iter_sample(MAX_N, steps, start=init_trace[0])

        time_grid_s = 1e-3 * time_grid_ms
        TC = time_chunker(sample_generator, time_grid_s, timers, n_grid=n_grid)

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

    # TODO remove test only
    reset_counters()
    with pm.Model():
        pm.DensityDist('x', logpdf, shape=D)
        steps = BUILD_STEP[sampler]()
        print 'doing offline run'
        trace_offline = pm.sample(len(trace), steps, init='advi',
                                  start={'x': params_dict[DATA_CENTER]})

    # TODO test only remove
    X_offline = format_trace(trace_offline)
    X_online = trace
    print 'offline'
    moments_report(X_offline)
    print 'online'
    moments_report(X_online)
    scaler = StandardScaler()
    X_exact = scaler.fit_transform(X_exact)
    X_offline = scaler.transform(X_offline)
    X_online = scaler.transform(X_online)
    print 'offline %f' % np.mean((np.mean(X_exact, axis=0) - np.mean(X_offline, axis=0)) ** 2)
    print 'online %f' % np.mean((np.mean(X_exact, axis=0) - np.mean(X_online, axis=0)) ** 2)

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
    assert(sampler == config['exact_name'] or sampler in BUILD_STEP)

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
                             config['t_grid_ms'], config['n_grid'])
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
