# Ryan Turner (turnerry@iro.umontreal.ca)
import sys
import fileio as io
from main import run_experiment


def main():
    # We would print usage error instead if we wanted to be user friendly
    assert(len(sys.argv) == 2)
    config_file = io.abspath2(sys.argv[1])

    print 'config %s' % config_file
    config = io.load_config(config_file)

    chains = io.get_chains(config['input_path'], config['csv_ext'])
    print 'inputs chains:'
    print chains

    for mc_chain_name in chains:
        run_experiment(config, mc_chain_name)
    print 'done'

if __name__ == '__main__':
    main()
