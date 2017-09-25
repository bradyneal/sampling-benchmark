# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import ConfigParser
import os
import sys
from tempfile import NamedTemporaryFile
import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3.backends.tracetab import trace_to_dataframe
import theano
from models import BUILD_MODEL, SAMPLE_MODEL
from samplers import BUILD_STEP
from chunker import time_chunker
from chunker import CHUNK_SIZE, GRID_INDEX
# These modules should be replaced with better options if phase3 goes Python3
from time import time as wall_time
from time import clock as cpu_time

SAMPLE_INDEX_COL = 'sample'
DATA_CENTER = 'data_center'
DATA_SCALE = 'data_scale'
DATA_EXT = '.csv'
MAX_N = 1000000  # Some value to prevent blowing out HDD space with samples.

# ============================================================================
# TODO move everything here to general util file


def format_trace(trace):
    # TODO I don't think we need to import extra function to get df from trace
    df = trace_to_dataframe(trace)
    return df.values


def abspath2(fname):
    return os.path.abspath(os.path.expanduser(fname))


def is_safe_name(name_str, sep_chars='_-'):
    safe = name_str.translate(None, sep_chars).isalnum()
    return safe


def build_output_name(param_name, sampler, sep='_', sub_sep='-'):
    output_name = ''.join((param_name, sep, sampler, sub_sep))
    assert(is_safe_name(output_name))
    return output_name


def get_temp_filename(dir_, prefix, ext=DATA_EXT):
    # Warning! NamedTemporaryFile sometimes put _ in the random string, we need
    # to be careful this does not mess up our convention. There does not appear
    # to be a way to stop this.
    tf = NamedTemporaryFile(suffix=ext, prefix=prefix, dir=dir_, delete=False)
    fname = tf.name
    tf.close()
    assert(os.path.isabs(fname))
    return fname

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


def load_config(config_file):
    config = ConfigParser.RawConfigParser()
    assert(os.path.isabs(config_file))
    config.read(config_file)

    input_path = abspath2(config.get('phase2', 'output_path'))
    output_path = abspath2(config.get('phase3', 'output_path'))
    t_grid_ms = config.getint('phase3', 'time_grid_ms')
    n_grid = config.getint('phase3', 'n_grid')
    n_exact = config.getint('phase3', 'n_exact')
    assert(n_exact > 0)  # didn't test =0 case

    pkl_ext = config.get('common', 'pkl_ext')
    meta_ext = config.get('common', 'meta_ext')
    exact_name = config.get('common', 'exact_name')
    assert(exact_name.isalnum())

    csv_ext = config.get('common', 'csv_ext')
    assert(csv_ext == DATA_EXT)  # For now just assert instead of pass

    # TODO get dict
    return input_path, output_path, pkl_ext, meta_ext, exact_name, t_grid_ms, n_grid, n_exact


def controller(model_setup, sampler, time_grid_ms, n_grid):
    assert(sampler in BUILD_STEP)
    assert(time_grid_ms > 0)

    model_name, D, params_dict = model_setup
    assert(model_name in BUILD_MODEL)

    timers = [('chunk_cpu_time_s', cpu_time),
              ('chunk_wall_time_s', wall_time),
              ('energy_calls', get_counters)]

    print 'starting experiment'
    print 'D=%d' % D
    assert(D >= 1)

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

    reset_counters()
    with pm.Model():
        pm.DensityDist('x', logpdf, shape=D, testval=np.zeros(D))
        steps = BUILD_STEP[sampler]()
        sample_generator = pm.sampling.iter_sample(MAX_N, steps)

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


def run_experiment(config, param_name, sampler):
    input_path, output_path, pkl_ext, meta_ext, exact_name, t_grid_ms, n_grid, n_exact = config

    assert(sampler == exact_name or sampler in BUILD_STEP)

    model_file = os.path.join(input_path, param_name + pkl_ext)
    print 'loading %s' % model_file
    assert(os.path.isabs(model_file))
    with open(model_file, 'rb') as f:
        model_setup = pkl.load(f)
    model_name, D, params_dict = model_setup
    assert(model_name in SAMPLE_MODEL)

    # Reserve temp file to avoid collisions
    data_file_name = build_output_name(param_name, sampler)
    data_file_name = get_temp_filename(output_path, data_file_name)

    # Now sample
    if sampler == exact_name:
        X = SAMPLE_MODEL[model_name](params_dict, N=n_exact)
        assert(X.shape == (n_exact, D))
        # Benchmark model trained on standardized data, move back to original.
        X = params_dict[DATA_SCALE][None, :] * X + \
            params_dict[DATA_CENTER][None, :]
    else:
        X, meta = controller(model_setup, sampler, t_grid_ms, n_grid)

        # Save meta data
        meta_file_name = data_file_name + meta_ext
        print 'saving meta-data to %s' % meta_file_name
        assert(not os.path.isfile(meta_file_name))  # This could be warning
        assert(not meta.isnull().any().any())
        meta.to_csv(meta_file_name, header=True, index=False)
    # Now save the data
    print 'saving samples to %s' % data_file_name
    np.savetxt(data_file_name, X, delimiter=',')


def main():
    # Could use a getopt package if this got fancy, but this is simple enough
    assert(len(sys.argv) == 4)
    config_file = abspath2(sys.argv[1])
    param_name = sys.argv[2]
    sampler = sys.argv[3]
    # TODO add option to control random seed
    assert(is_safe_name(param_name))

    config = load_config(config_file)

    run_experiment(config, param_name, sampler)
    print 'done'

if __name__ == '__main__':
    main()
