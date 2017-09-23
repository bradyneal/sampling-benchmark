# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import ConfigParser
import os
import sys
from tempfile import mkdtemp
import numpy as np
from sklearn.preprocessing import StandardScaler
from model_wrappers import STD_BENCH_MODELS

# Currently first requires:
# export PYTHONPATH=./bench_models/nade/:$PYTHONPATH
# TODO fix, i don't like that, need to fix some garbage in __init__.py files

PHASE3_MODELS = ('MoG', 'RNADE')
DATA_EXT = '.csv'
abspath2 = os.path.abspath  # TODO write combo func here


def build_output_name(mc_chain_name, model_name, pkl_ext):
    output_name = '%s_%s%s' % (mc_chain_name, model_name, pkl_ext)
    return output_name


def is_safe_name(name_str):
    # TODO make extra chars configurable
    safe = name_str.translate(None, '-_').isalnum()
    return safe


def load_config(config_file):
    config = ConfigParser.RawConfigParser()
    assert(os.path.isabs(config_file))
    config.read(config_file)

    input_path = abspath2(config.get('phase1', 'output_path'))
    output_path = abspath2(config.get('phase2', 'output_path'))
    pkl_ext = config.get('common', 'pkl_ext')

    csv_ext = config.get('common', 'csv_ext')
    assert(csv_ext == DATA_EXT)  # For now just assert instead of pass

    train_frac = config.getfloat('phase2', 'train_frac')
    assert(0.0 <= train_frac and train_frac <= 1.0)

    rnade_scratch = abspath2(config.get('phase2', 'rnade_scratch_dir'))
    # This is kind of the limit before we need to change to a dict or something
    return input_path, output_path, pkl_ext, train_frac, rnade_scratch


def get_default_run_setup(config):
    input_path, output_path, pkl_ext, train_frac, rnade_scratch = config
    # Can run multiple instances in parallel with random subdir for scratch
    rnade_scratch = mkdtemp(dir=rnade_scratch)
    print 'rnade scratch %s' % rnade_scratch
    assert(os.path.isabs(rnade_scratch))

    run_config = \
        {'norm_diag': ('MoG', {'n_components': 1, 'covariance_type': 'diag'}),
         'norm_full': ('MoG', {'n_components': 1, 'covariance_type': 'full'}),
         'MoG': ('MoG', {'n_components': 5}),
         'IGN': ('IGN', {'n_layers': 3, 'n_epochs': 2500, 'lr': 1e-4}),
         'RNADE': ('RNADE', {'n_components': 5, 'scratch_dir': rnade_scratch})}
    return run_config


def run_experiment(config, mc_chain_name, standardize=True, debug_dump=False,
                   setup=get_default_run_setup):
    '''Call this instead of main for scripted multiple runs within python.'''
    input_path, output_path, pkl_ext, train_frac, rnade_scratch = config

    mc_chain_file = os.path.join(input_path, mc_chain_name + DATA_EXT)
    print 'loading %s' % mc_chain_file

    run_config = setup(config)

    # TODO general util can include basic csv load and write
    assert(os.path.isabs(mc_chain_file))
    MC_chain = np.genfromtxt(mc_chain_file, dtype=float, delimiter=',',
                             skip_header=0, loose=False, invalid_raise=True)
    N, D = MC_chain.shape
    print 'size %d x %d' % (N, D)
    N_train = int(np.ceil(train_frac * N))

    best_loglik = -np.inf
    best_case = None
    model_dump = {}
    for run_name, (model_name, args) in run_config.iteritems():
        print 'running %s with arguments' % model_name
        # TODO use pretty print or whatever it is called to print dict nicely
        print args

        # leave at default params for now, can use fancy skopt stuff later.
        # All models are setup in sklearn pattern to make later use with skopt
        # easier, and can use sklearn estimators with no wrappers.
        model = STD_BENCH_MODELS[model_name](**args)
        model.fit(MC_chain[:N_train, :])
        # Get score for each sample, can then use benchmark tools for table
        # with error bars and all that at a later point.
        loglik_vec = model.score_samples(MC_chain[N_train:, :])

        params_obj = model.get_params()
        loglik_vec_chk = model.loglik_chk(MC_chain[N_train:, :], params_obj)
        err = np.max(np.abs(loglik_vec - loglik_vec_chk))
        print 'loglik chk log10 err %f' % np.log10(err)

        test_loglik = np.mean(loglik_vec)
        print '%s: %f' % (run_name, test_loglik)
        if model_name in PHASE3_MODELS and test_loglik > best_loglik:
            best_loglik = test_loglik
            best_case = (model_name, model)

        model_dump[run_name] = (model_name, params_obj)
    assert(best_case is not None)

    model_name, model = best_case
    print 'using %s' % model_name

    # There exist methods to pickle sklearn learns object, but these systems
    # seem brittle.  We also need to re-implement these objects anyway for
    # reuse with pymc3, so we might as well just save the parameters clean.
    params_obj = model.get_params()

    # TODO sample data here and do sanity check against original
    # Might make more sense just ot sample example here
    # on each time print: moments, test stat (t or U, BF, KS), p-val for test

    # Now dump to finish the job
    dump_file = build_output_name(mc_chain_name, model_name, pkl_ext)
    dump_file = os.path.join(output_path, dump_file)
    print 'saving %s' % dump_file
    assert(os.path.isabs(dump_file))
    with open(dump_file, 'wb') as f:
        pkl.dump((model_name, D, params_obj), f, pkl.HIGHEST_PROTOCOL)

    # Also dump everything in another pkl file for debug purposes
    if debug_dump:
        dump_file = os.path.join(output_path, 'all_model_dump') + pkl_ext
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
    config_file = abspath2(sys.argv[1])
    mc_chain_name = sys.argv[2]
    assert(is_safe_name(mc_chain_name))

    print 'config %s' % config_file
    config = load_config(config_file)

    run_experiment(config, mc_chain_name)
    print 'done'

if __name__ == '__main__':
    main()
