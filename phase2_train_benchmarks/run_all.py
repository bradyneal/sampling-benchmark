#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)
import sys
from joblib import Parallel, delayed
from functools import partial
import fileio as io
from main import run_experiment
import traceback


def main():
    num_args = len(sys.argv) - 1
    if num_args < 1:
        config_path = '../config.ini'
    elif num_args > 1:
        raise Exception('too many arguments: %d. %d expected' % (num_args, 1))
    else:
        config_path = sys.argv[1]
    config_file = io.abspath2(config_path)

    print 'config %s' % config_file
    config = io.load_config(config_file)
    print(config['input_path'])

    chains = io.get_chains(config['input_path'], config['csv_ext'],
                           config['size_limit_bytes'])
    print 'inputs chains:'
    print chains

    print 'Running njobs=%d in parallel' % config['njobs']
    try_run_experiment_with_config = partial(try_run_experiment, config)
    Parallel(n_jobs=config['njobs'])(
        map(delayed(try_run_experiment_with_config), chains))
    print 'done'


def try_run_experiment(config, mc_chain_name):
    """
    Put run_experiment inside a try-except so that one failed experiment
    doesn't kill them all when many are running in parallel.
    """
    try:
        run_experiment(config, mc_chain_name)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        print '%s failed' % mc_chain_name
        traceback.print_exc()

if __name__ == '__main__':
    main()
