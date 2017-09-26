# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import os
import sys
from tempfile import mkdtemp
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import fileio as io
from model_wrappers import STD_BENCH_MODELS

# Currently first requires:
# export PYTHONPATH=./bench_models/nade/:$PYTHONPATH
# TODO fix, i don't like that, need to fix some garbage in __init__.py files

PHASE3_MODELS = ('MoG', 'VBMoG', 'RNADE')  # Models implemented in phase 3
DATA_CENTER = 'data_center'
DATA_SCALE = 'data_scale'


def get_default_run_setup(config):
    max_mixtures = 25

    # Can run multiple instances in parallel with random subdir for scratch
    rnade_scratch = mkdtemp(dir=config['rnade_scratch'])
    print 'rnade scratch %s' % rnade_scratch
    assert(os.path.isabs(rnade_scratch))

    run_config = \
        {'norm_diag': ('Gaussian', {'diag': True}, {}),
         'norm_full': ('Gaussian', {'diag': False}, {}),
         'MoG': ('MoG', {}, {'n_components': range(2, max_mixtures + 1)}),
         'VBMoG': ('VBMoG', {'n_components': max_mixtures}, {}),
         'IGN': ('IGN', {'n_layers': 3, 'n_epochs': 2500, 'lr': 1e-4}, {}),
         'RNADE': ('RNADE',
                   {'n_components': 5, 'scratch_dir': rnade_scratch}, {})}
    return run_config


def run_experiment(config, chain_name, standardize=True, debug_dump=False,
                   setup=get_default_run_setup, shuffle=False):
    '''Call this instead of main for scripted multiple runs within python.'''
    run_config = setup(config)

    MC_chain = io.load_np(config['input_path'], chain_name, config['csv_ext'])
    N, D = MC_chain.shape
    print 'size %d x %d' % (N, D)
    N_train = int(np.ceil(config['train_frac'] * N))

    # Shuffle to make train/test look alike since chain is not iid data. Maybe
    # we should thin before a shuffle if the input chain is poorly mixing.
    if shuffle:
        print 'Doing shuffle! Consider thinning!'
        np.random.shuffle(MC_chain)
    X_train, X_test = MC_chain[:N_train, :], MC_chain[N_train:, :]

    if standardize:
        # We can go to robust scaler if we still have trouble
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    best_loglik = -np.inf
    best_case = None
    model_dump = {}
    for run_name, (model_name, args, cv_args) in run_config.iteritems():
        print 'running %s with arguments' % model_name
        print args
        print 'grid searching over parameters'
        print cv_args.keys()

        # Do the training
        model = STD_BENCH_MODELS[model_name](**args)
        if len(cv_args) == 0:
            model.fit(X_train)
        else:
            # If the optimization require a larger space we can switch to skopt
            cv_model = GridSearchCV(model, cv_args)
            cv_model.fit(X_train)
            model = cv_model.best_estimator_  # Get original model back out
            print 'CV optimum'
            print cv_model.best_params_
        # Get the performance for this model
        params_obj = model.get_params_()
        # Using _chk function as the real b.c. that is what we use in phase 3
        loglik_vec = model.loglik_chk(X_test, params_obj)
        test_loglik = np.mean(loglik_vec)
        print 'loglik %s: %f' % (run_name, test_loglik)

        # Check we get same answer as sklearn built in score
        loglik_vec_chk = model.score_samples(X_test)
        err = np.max(np.abs(loglik_vec - loglik_vec_chk))
        print 'loglik chk %s log10 err: %f' % (run_name, np.log10(err))
        print 'loglik chk %s: %f' % (run_name, np.mean(loglik_vec_chk))

        # Update which is best so far
        if model_name in PHASE3_MODELS and test_loglik > best_loglik:
            best_loglik = test_loglik
            best_case = (model_name, model)
        model_dump[run_name] = (model_name, params_obj)  # For debug dump
    assert(best_case is not None)

    model_name, model = best_case
    print 'using %s' % model_name

    # There exist methods to pickle sklearn learns object, but these systems
    # seem brittle.  We also need to re-implement these objects anyway for
    # reuse with pymc3, so we might as well just save the parameters clean.
    params_obj = model.get_params_()
    # Save the scale info to get back to the original data space too.
    assert(DATA_CENTER not in params_obj)
    assert(DATA_SCALE not in params_obj)
    params_obj[DATA_CENTER] = scaler.mean_
    params_obj[DATA_SCALE] = scaler.scale_

    # Now dump to finish the job
    dump_file = io.build_output_name(chain_name, model_name, config['pkl_ext'])
    dump_file = os.path.join(config['output_path'], dump_file)
    print 'saving %s' % dump_file
    assert(os.path.isabs(dump_file))
    with open(dump_file, 'wb') as f:
        pkl.dump((model_name, D, params_obj), f, pkl.HIGHEST_PROTOCOL)

    # Also dump everything in another pkl file for debug purposes
    if debug_dump:
        dump_file = 'all_model_dump' + config['pkl_ext']
        dump_file = os.path.join(config['output_path'], dump_file)
        print 'saving %s' % dump_file
        assert(os.path.isabs(dump_file))
        with open(dump_file, 'wb') as f:
            pkl.dump(model_dump, f, pkl.HIGHEST_PROTOCOL)
    return model_dump


def main():
    '''This program can be run in parallel across different MC_chain files
    indep. This is a top level routine so I am not worried about needing a
    verbosity setting.'''
    assert(len(sys.argv) == 3)  # Print usage error instead to be user friendly
    config_file = io.abspath2(sys.argv[1])
    mc_chain_name = sys.argv[2]
    assert(io.is_safe_name(mc_chain_name))

    print 'config %s' % config_file
    config = io.load_config(config_file)

    run_experiment(config, mc_chain_name)
    print 'done'

if __name__ == '__main__':
    main()
