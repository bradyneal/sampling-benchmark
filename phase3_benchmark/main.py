# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import csv
import ConfigParser
import os
import sys
from tempfile import mkstemp
import numpy as np
import pymc3 as pm
from pymc3.backends.tracetab import trace_to_dataframe
import theano
from models import BUILD_MODEL, SAMPLE_MODEL
from samplers import BUILD_STEP
# These modules should be replaced with better options if phase3 goes Python3
from time import time as wall_time
from time import clock as cpu_time

DATA_EXT = '.csv'
FILE_FMT = '%s_%s-'  # Add - before random tempfile string
MAX_N = 1000000

abspath2 = os.path.abspath  # TODO write combo func here

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

    pkl_ext = config.get('common', 'pkl_ext')
    meta_ext = config.get('common', 'meta_ext')
    exact_name = config.get('common', 'exact_name')

    csv_ext = config.get('common', 'csv_ext')
    assert(csv_ext == DATA_EXT)  # For now just assert instead of pass

    return input_path, output_path, pkl_ext, meta_ext, exact_name, t_grid_ms, n_grid


def format_trace(trace):
    # TODO I don't think we need to import extra function to get df from trace
    df = trace_to_dataframe(trace)
    return df.values


def is_safe_name(name_str, allow_dot=False):
    # TODO make extra chars configurable
    ignore = '-_.' if allow_dot else '-_'
    safe = name_str.translate(None, ignore).isalnum()
    return safe


def next_grid(x, grid_size):
    grid_idx = int(x // grid_size) + 1
    x_next = grid_size * grid_idx
    assert(x_next > x)
    assert(x_next - x <= grid_size)  # no skipping
    return grid_idx, x_next


def controller(model_setup, sampler, data_f, meta_f, time_grid_ms, n_grid):
    assert(sampler in BUILD_STEP)
    assert(time_grid_ms > 0)

    model_name, D, params_dict = model_setup
    assert(model_name in BUILD_MODEL)

    meta_headers = ['grid_idx', 'sample',
                    'grid_time_ms', 'chunk_cpu_time_s', 'chunk_wall_time_s',
                    'chunk_size', 'f_calls']
    meta_writer = csv.writer(meta_f)
    meta_writer.writerow(meta_headers)

    print 'starting experiment'
    print 'D=%d' % D
    assert(D >= 1)

    # Use default arg trick to get params to bind to model now
    def logpdf(x, p=params_dict):
        # This is Fred's trick to implicitly count function evals in theano.
        s = theano.shared(0, name='function_calls')
        all_counters.append(s)
        s.default_update = s + 1
        ll = BUILD_MODEL[model_name](x, p)
        return ll + s * 0

    # TODO put all this in some form of generator, it would be cleaner
    # Can also just build up meta-data with generator, and skip file IO
    # Incremental file IO is not really needed here

    reset_counters()
    last_count = 0

    total_cpu_time = 0.0
    chunk_cpu_time = 0.0
    chunk_wall_time = 0.0

    grid_idx = 0
    next_dump_time_ms = 0.0

    time_dbg_chk = 0.0
    len_chk = 0
    with pm.Model():
        pm.DensityDist('x', logpdf, shape=D, testval=np.zeros(D))
        steps = BUILD_STEP[sampler]()
        sample_generator = pm.sampling.iter_sample(MAX_N, steps)

        print 'setup function calls:'
        print get_counters()

        print 'starting to sample'
        # TODO figure out where init time is and dump into meta_f
        last_chunk = 0
        for ii in xrange(MAX_N):
            grid_chk, dump_time_ms_chk = next_grid(1e3 * total_cpu_time, time_grid_ms)

            # Get calls not including this point, is there much overhead to
            # doing this each iter??
            f_calls = get_counters()
            assert(np.ndim(f_calls) == 0)

            tc, tw = cpu_time(), wall_time()
            trace = next(sample_generator)
            tc_delta, tw_delta = cpu_time() - tc, wall_time() - tw

            # TODO dump to dbg each iter

            total_cpu_time += tc_delta
            # Might be safest to convert this to int as well but need to think
            # about if we want floor or ceil.
            total_cpu_time_ms = 1e3 * total_cpu_time
            if total_cpu_time_ms > next_dump_time_ms:
                assert(ii == len(trace) - 1)  # sanity check
                chunk_size = ii - last_chunk
                # TODO move check inside format trace
                X = np.zeros((0, D)) if chunk_size == 0 else \
                    format_trace(trace[last_chunk:ii])
                last_chunk = ii
                assert(X.shape == (chunk_size, D))

                # Write out data, np can handle chunk_size==0 fine it seems
                np.savetxt(data_f, X, delimiter=',')

                time_dbg_chk += chunk_cpu_time
                # TODO check is approx equal to next grid on chunk_cpu_time
                assert(time_dbg_chk <= next_dump_time_ms)
                assert(ii == 0 or next_dump_time_ms == dump_time_ms_chk)
                assert(ii == 0 or grid_idx == grid_chk)
                len_chk += chunk_size
                assert(ii == len_chk)

                # Write out meta-data
                chunk_calls = f_calls - last_count
                last_count = f_calls
                meta = [grid_idx, ii,
                        next_dump_time_ms, chunk_cpu_time, chunk_wall_time,
                        chunk_size, chunk_calls]
                print meta
                meta_writer.writerow(meta)
                chunk_cpu_time, chunk_wall_time = 0.0, 0.0

                # Find the next grid point for quantized time
                grid_idx, next_dump_time_ms = \
                    next_grid(total_cpu_time_ms, time_grid_ms)
                if grid_idx >= n_grid:
                    break
            chunk_cpu_time += tc_delta
            chunk_wall_time += tw_delta
        print 'function calls:'
        print get_counters()
    sec_sample = total_cpu_time / len(trace)
    print 's/sample = %f' % sec_sample
    return sec_sample


def run_experiment(config, param_name, sampler, n_exact):
    input_path, output_path, pkl_ext, meta_ext, exact_name, t_grid_ms, n_grid = config

    assert(sampler == exact_name or sampler in BUILD_STEP)

    model_file = os.path.join(input_path, param_name + pkl_ext)
    print 'loading %s' % model_file
    assert(os.path.isabs(model_file))
    with open(model_file, 'rb') as f:
        model_setup = pkl.load(f)
    model_name, D, params_dict = model_setup
    assert(model_name in SAMPLE_MODEL)

    sample_file = FILE_FMT % (param_name, sampler)
    # TODO put following in util, the suffix not really needed on exact, but
    # let's do it for consistency
    assert(is_safe_name(sample_file))
    data_f, data_path = mkstemp(suffix=DATA_EXT, prefix=sample_file,
                                dir=output_path, text=False)
    data_f = os.fdopen(data_f, 'ab')  # Convert to actual file object
    print 'saving samples to %s' % data_path

    if sampler == exact_name:
        X = SAMPLE_MODEL[model_name](params_dict, N=n_exact)
        assert(X.shape == (n_exact, D))
        np.savetxt(data_f, X, delimiter=',')
    else:
        meta_file_name = data_path + meta_ext
        print 'saving meta-data to %s' % meta_file_name
        assert(not os.path.isfile(meta_file_name))  # This could be warning
        with open(meta_file_name, 'ab') as meta_f:  # Must use append mode!
            controller(model_setup, sampler, data_f, meta_f, t_grid_ms, n_grid)
    data_f.close()


def main():
    # Could use a getopt package if this got fancy, but this is simple enough
    assert(len(sys.argv) == 5)
    config_file = abspath2(sys.argv[1])
    param_name = sys.argv[2]
    sampler = sys.argv[3]
    n_exact = int(sys.argv[4])
    # TODO add option to control random seed

    assert(is_safe_name(param_name))
    assert(n_exact >= 0)

    config = load_config(config_file)

    run_experiment(config, param_name, sampler, n_exact)
    print 'done'  # Job will probably get killed before we get here.

if __name__ == '__main__':
    main()
