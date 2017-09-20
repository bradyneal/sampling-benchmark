# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import ConfigParser
import os
import sys
from time import time
import numpy as np
import pymc3 as pm
from pymc3.backends.tracetab import trace_to_dataframe
from models import BUILD_MODEL, SAMPLE_MODEL
from samplers import BUILD_STEP

DATA_EXT = '.csv'
FILE_FMT = '%s_%s%s'

abspath2 = os.path.abspath  # TODO write combo func here


def load_config(config_file):
    config = ConfigParser.RawConfigParser()
    assert(os.path.isabs(config_file))
    config.read(config_file)

    input_path = abspath2(config.get('phase2', 'output_path'))
    output_path = abspath2(config.get('phase3', 'output_path'))

    pkl_ext = config.get('common', 'pkl_ext')
    exact_name = config.get('common', 'exact_name')

    csv_ext = config.get('common', 'csv_ext')
    assert(csv_ext == DATA_EXT)  # For now just assert instead of pass

    return input_path, output_path, pkl_ext, exact_name


def format_trace(trace):
    # TODO I don't think we need to import extra function to get df from trace
    df = trace_to_dataframe(trace)
    return df.values


def is_safe_name(name_str, allow_dot=False):
    # TODO make extra chars configurable
    ignore = '-_.' if allow_dot else '-_'
    safe = name_str.translate(None, ignore).isalnum()
    return safe


def sampling_part(model_name, D, params_dict, sampler, outfile_f, max_N):
    print 'starting experiment'
    print 'D=%d' % D
    assert(D >= 1)

    # Use default arg trick to get params to bind to model now
    logpdf = lambda x, p=params_dict: BUILD_MODEL[model_name](x, p)

    with pm.Model():
        pm.DensityDist('x', logpdf, shape=D, testval=np.zeros(D))
        steps = BUILD_STEP[sampler]()
        sample_generator = pm.sampling.iter_sample(max_N, steps)

        print 'starting to sample'
        # TODO somehow log timing information in a file
        for ii in xrange(max_N):
            # TODO consider clock vs time
            t = time()
            trace = next(sample_generator)
            t = time() - t

            X = format_trace(trace[-1:])
            # TODO be nice if we could figure out how to do this chunkwise
            assert(X.shape == (1, D))
            # TODO make sure to flush every once in a while
            np.savetxt(outfile_f, X, delimiter=',')
            print t
    return


def run_experiment(config, param_name, sampler, max_N):
    input_path, output_path, pkl_ext, exact_name = config

    assert(sampler == exact_name or sampler in BUILD_STEP)

    model_file = os.path.join(input_path, param_name + pkl_ext)
    print 'loading %s' % model_file
    assert(os.path.isabs(model_file))
    with open(model_file, 'rb') as f:
        model_name, D, params_dict = pkl.load(f)
    assert(model_name in SAMPLE_MODEL)

    sample_file = FILE_FMT % (param_name, sampler, DATA_EXT)
    assert(is_safe_name(sample_file, allow_dot=True))
    sample_file = os.path.join(output_path, sample_file)

    # We could move the open and close inside run_experiment() to not keep an
    # extra file handle open, but whatever, this is good enough for now.
    # TODO add warning if file aready exists
    assert(os.path.isabs(sample_file))
    with open(sample_file, 'ab') as f:  # Critical to use append mode!
        if sampler == exact_name:
            X = SAMPLE_MODEL[model_name](params_dict, N=max_N)
            assert(X.shape == (max_N, D))
            np.savetxt(f, X, delimiter=',')
        else:
            sampling_part(model_name, D, params_dict, sampler, f, max_N=max_N)
    return


def main():
    # Could use a getopt package if this got fancy, but this is simple enough
    assert(len(sys.argv) == 5)
    config_file = abspath2(sys.argv[1])
    param_name = sys.argv[2]
    sampler = sys.argv[3]
    max_N = int(sys.argv[4])
    # TODO add option to control random seed

    assert(is_safe_name(param_name))
    assert(max_N >= 0)

    config = load_config(config_file)

    run_experiment(config, param_name, sampler, max_N)
    print 'done'  # Job will probably get killed before we get here.

if __name__ == '__main__':
    main()
