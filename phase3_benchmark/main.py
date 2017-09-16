# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import os
import sys
import numpy as np
import pymc3 as pm
from pymc3.backends.tracetab import trace_to_dataframe
from models import BUILD_MODEL
from samplers import BUILD_STEP

DATA_EXT = '.csv'  # TODO go into config

CHUNK_SIZE = 1000  # Could make this command arg if we like
FILE_FMT = '%s_%s' + DATA_EXT


def chomp(ss, ext):
    L = len(ext)
    assert(ss[-L:] == ext)
    return ss[:-L]


def get_real_base(fname, ext):
    _, fname = os.path.split(fname)
    return chomp(fname, ext)


def format_trace(trace):
    # TODO I don't think we need to import extra function to get df from trace
    df = trace_to_dataframe(trace)
    return df.values


def run_experiment(model_name, params_dict, sampler, outfile_f, max_N=np.inf):
    print 'starting experiment'

    # TODO save D in pkl file itself
    if model_name == 'RNADE':
        D = len(params_dict['orderings'][0])
    elif model_name == 'MoG':
        D = params_dict['means'].shape[1]
    else:
        assert(False)
    print 'D=%d' % D

    # Use default arg trick to get params to bind to model now
    logpdf = lambda x, p=params_dict: BUILD_MODEL[model_name](x, p)

    with pm.Model():
        pm.DensityDist('x', logpdf, shape=D, testval=np.zeros(D))
        steps = BUILD_STEP[sampler]()
        sample_generator = pm.sampling.iter_sample(CHUNK_SIZE, steps)

        print 'starting to sample'
        # TODO somehow log timing information in here
        N = 0
        for trace in sample_generator:
            # TODO control some sort of thinning here
            X = format_trace(trace[-1:])
            np.savetxt(outfile_f, X, delimiter=',')
            print X.shape
            # assert(X.ndim == 2 and X.shape[0] == CHUNK_SIZE)

            # Could multiply enumerate counter by chunk size, but both are simple
            N += CHUNK_SIZE
            if N >= max_N:
                break
    return N


def main():
    # Could use a getopt package if this got fancy, but this is simple enough
    assert(len(sys.argv) in (3, 4))
    mc_chain_file = sys.argv[1]
    sampler = sys.argv[2]
    max_N = np.inf if len(sys.argv) <= 3 else int(sys.argv[3])

    # TODO all of these should go in config file
    input_path = '.'
    output_path = '.'
    pkl_ext = '.pkl'

    model_file = os.path.join(input_path, mc_chain_file) + pkl_ext
    print 'loading %s' % model_file
    with open(model_file, 'rb') as f:
        model_name, params_dict = pkl.load(f)

    sample_file = FILE_FMT % (get_real_base(mc_chain_file, DATA_EXT), sampler)
    # TODO verify sample_file safe, e.g. no \ or / etc
    sample_file = os.path.join(output_path, sample_file)

    # We could move the open and close inside run_experiment() to not keep an
    # extra file handle open, but whatever, this is good enough for now.
    with open(sample_file, 'ab') as f:  # Critical to use append mode!
        run_experiment(model_name, params_dict, sampler, f, max_N=max_N)
    print 'done'  # Job will probably get killed before we get here.

if __name__ == '__main__':
    main()
