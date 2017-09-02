# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle
import os
import sys
import numpy as np
from model_wrappers import PARAM_EXTRACTORS, STD_BENCH_MODELS

# Currently first requires:
# export PYTHONPATH=./bench_models/nade/:$PYTHONPATH
# TODO fix, i don't like that


def main():
    '''This program can be run in parallel across different MC_chain files
    indep. This is a top level routine so I am not worried about needing a
    verbosity setting.'''
    # The config file might become another cmd line arg and might go at top
    # level since it will be shared across phases
    input_path = '.'  # TODO Get path for files from config file
    output_path = '.'  # TODO Get path for files from config file
    pkl_ext = '.pkl'  # TODO PKL_EXT goes in the config file too

    assert(len(sys.argv) == 2)  # Print usage error instead to be user friendly
    mc_chain_file = sys.argv[1]
    mc_chain_file = os.path.join(input_path, mc_chain_file)
    print 'loading %s' % mc_chain_file

    # Use np to directly load in csv, if this becomes a problem then use pandas
    # and then .values to get an np array out. Set to raise an error if
    # anything weird in file.
    MC_chain = np.genfromtxt(mc_chain_file, dtype=float, delimiter=',',
                             skip_header=0, loose=False, invalid_raise=True)
    print 'size %d x %d' % (MC_chain.shape[0], MC_chain.shape[1])
    # We could also print a scipy describe function or something to be fancy
    # TODO setup based on a fraction arg or something, maybe from config
    N_train = int(np.ceil(0.8 * MC_chain.shape[0]))

    best_loglik = -np.inf
    best_case = None
    model_dump = {}
    for model_name, bench_model in STD_BENCH_MODELS.iteritems():
        print 'running %s' % model_name

        # leave at default params for now, can use fancy skopt stuff later.
        # All models are setup in sklearn pattern to make later use with skopt
        # easier, and can use sklearn estimators with no wrappers.
        model = bench_model()
        model.fit(MC_chain[:N_train, :])
        # Get score for each sample, can then use benchmark tools for table
        # with error bars and all that at a later point.
        loglik_vec = model.score_samples(MC_chain[N_train:, :])

        test_loglik = np.mean(loglik_vec)
        print '%s: %f' % (model_name, test_loglik)
        if test_loglik > best_loglik:
            best_loglik = test_loglik
            best_case = (model_name, model)

        model_dump[model_name] = PARAM_EXTRACTORS[model_name](model)
    assert(best_case is not None)

    model_name, model = best_case
    print 'using %s' % model_name

    # There exist methods to pickle sklearn learns object, but these systems
    # seem brittle.  We also need to re-implement these objects anyway for
    # reuse with pymc3, so we might as well just save the parameters clean.
    params_obj = PARAM_EXTRACTORS[model_name](model)

    # Now dump to finish the job
    dump_file = os.path.join(output_path, mc_chain_file) + pkl_ext
    print 'saving %s' % dump_file
    with open(dump_file, 'wb') as f:
        cPickle.dump((model_name, params_obj), f, cPickle.HIGHEST_PROTOCOL)

    # Now dump to finish the job
    dump_file = os.path.join(output_path, 'all_model_dump') + pkl_ext
    print 'saving %s' % dump_file
    with open(dump_file, 'wb') as f:
        cPickle.dump(model_dump, f, cPickle.HIGHEST_PROTOCOL)
    print 'done'

if __name__ == '__main__':
    main()
