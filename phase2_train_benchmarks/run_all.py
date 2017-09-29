#!/usr/bin/env python
# Ryan Turner (turnerry@iro.umontreal.ca)
import sys
import fileio as io
from main import run_experiment


def main():
    num_args = len(sys.argv) - 1
    if num_args < 1:
        config_path = '../config.ini'
    elif len(sys.argv) > 1:
        raise Exception('too many arguments: %d. %d expected' % (num_args, 1))
    else:
        config_path = sys.argv[1]
    config_file = io.abspath2(config_path)

    print 'config %s' % config_file
    config = io.load_config(config_file)
    print(config['input_path'])

    chains = io.get_chains(config['input_path'], config['csv_ext'])
    print 'inputs chains:'
    print chains

    for mc_chain_name in chains:
        try:
            run_experiment(config, mc_chain_name)
        except:
            print '%s failed' % mc_chain_name
    print 'done'

if __name__ == '__main__':
    main()
