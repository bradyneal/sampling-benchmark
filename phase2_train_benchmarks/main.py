# Ryan Turner (turnerry@iro.umontreal.ca)
# Currently first requires:
# Either
# export PYTHONPATH=./bench_models/nade/:$PYTHONPATH
# or import __init__ (done)
# TODO fix, i don't like that, need to fix some garbage in __init__.py files

import __init__
import cPickle as pkl
import os
import sys
from tempfile import mkdtemp
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import fileio as io
from model_wrappers import STD_BENCH_MODELS
from validate_input_data import moments_report
import pandas as pd

PHASE3_MODELS = ('MoG', 'VBMoG', 'RNADE')  # Models implemented in phase 3
DATA_CENTER = 'data_center'
DATA_SCALE = 'data_scale'
META = 'meta'


def get_default_run_setup(config):
    max_mixtures = 25

    # Make rnade_scratch dir if doesn't already exist
    try:
        os.makedirs(config['rnade_scratch'])
    except OSError:
        if not os.path.isdir(config['rnade_scratch']):
            raise

    # Can run multiple instances in parallel with random subdir for scratch
    rnade_scratch = mkdtemp(dir=config['rnade_scratch'])
    print 'rnade scratch %s' % rnade_scratch
    assert(os.path.isabs(rnade_scratch))

    # 'IGN': ('IGN', {'n_layers': 3, 'n_epochs': 2500, 'lr': 1e-4}, {}),

    run_config = \
        {'norm_diag': ('Gaussian', {'diag': True}, {}),
         'norm_full': ('Gaussian', {'diag': False}, {}),
         'MoG': ('MoG', {}, {'n_components': range(2, max_mixtures + 1)}),
         'VBMoG': ('VBMoG', {'n_components': max_mixtures}, {}),
         'RNADE': ('RNADE',
                   {'n_components': 5, 'scratch_dir': rnade_scratch}, {})}
    return run_config


def use_model(model_name, args, cv_args, X_train, X_test):
    '''Anything that uses the model obj goes here to be in try-catch.'''
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
    # Check we get same answer as sklearn built in score
    loglik_vec_chk = model.score_samples(X_test)
    return model, params_obj, loglik_vec, loglik_vec_chk


def run_experiment(config, chain_name, debug_dump=False, shuffle=False,
                   setup=get_default_run_setup):
    '''Call this instead of main for scripted multiple runs within python.'''
    run_config = setup(config)
    # TODO remove: new phase 1 already takes out burn in
    burn_in_frac = 0.05

    chain_df = io.load_df(config['input_path'], chain_name, config['csv_ext'])
    
    # drop redundant columns according to Fisher information
    if config['drop_redundant_cols']:
        max_scale = io.load_fisher_info(config['input_path'], chain_name)
        print 'max_scale:'
        print max_scale
        if len(chain_df.columns) != max_scale.shape[0] or max_scale.ndim != 1:
            print 'Incorrect number of columns!!!! Not dropping any columns.'
            print 'shape of max_scale', max_scale.shape
            print 'num actual columns:', len(chain_df.columns)
        else:
            chain_df = chain_df.iloc[:, max_scale > config['max_scale_epsilon']]
            if chain_df.empty:
                print 'All columns redundant! Moving on...'
                assert(False)
        
    MC_chain = pd.DataFrame.as_matrix(chain_df)
    
    print 'full'
    moments_report(MC_chain)

    MC_chain = MC_chain[int(burn_in_frac * MC_chain.shape[0]):, :]
    print 'post burn-in'
    moments_report(MC_chain)

    N, D = MC_chain.shape
    print 'size %d x %d' % (N, D)
    N_train = int(np.ceil(config['train_frac'] * N))

    # Shuffle to make train/test look alike since chain is not iid data. Maybe
    # we should thin before a shuffle if the input chain is poorly mixing.
    if shuffle:
        print 'Doing shuffle! Consider thinning!'
        np.random.shuffle(MC_chain)
    X_train, X_test = MC_chain[:N_train, :], MC_chain[N_train:, :]

    # We can go to robust scaler if we still have trouble
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # I don't trust sklearn, so pull out values and validate to be sure.
    mean_, scale_ = scaler.mean_, scaler.scale_
    assert(np.allclose(X_train,
        (MC_chain[:N_train, :] - mean_[None, :]) / scale_[None, :]))
    assert(np.allclose(X_test,
        (MC_chain[N_train:, :] - mean_[None, :]) / scale_[None, :]))

    best_loglik = -np.inf
    best_case = None
    model_dump = {}
    for run_name, (model_name, args, cv_args) in run_config.iteritems():
        print 'running %s with arguments' % model_name
        print args
        print 'grid searching over parameters'
        print cv_args.keys()

        try:
            R = use_model(model_name, args, cv_args, X_train, X_test)
        except Exception as err:
            print '%s/%s failed' % (run_name, model_name)
            print str(err)
            continue
        model, params_obj, loglik_vec, loglik_vec_chk = R

        err = np.max(np.abs(loglik_vec - loglik_vec_chk))
        print 'loglik chk %s log10 err: %f' % (run_name, np.log10(err))

        print 'loglik chk %s: %f' % (run_name, np.mean(loglik_vec_chk))

        test_loglik = np.mean(loglik_vec)
        print 'loglik %s: %f' % (run_name, test_loglik)

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
    params_obj[DATA_CENTER] = mean_
    params_obj[DATA_SCALE] = scale_

    # Now build a meta-data dictionary
    meta = {headers: chain_df.columns.values.tolist()}
    print 'saving meta information:'
    print meta
    params_obj[META] = meta

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
