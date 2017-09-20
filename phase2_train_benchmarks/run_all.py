# Ryan Turner (turnerry@iro.umontreal.ca)
import os
import sys
import main as m


def chomp(ss, ext):
    L = len(ext)
    assert(ss[-L:] == ext)
    return ss[:-L]


def get_chains(input_path, ext=m.DATA_EXT):
    chains = sorted(chomp(fname, ext) for fname in os.listdir(input_path))
    return chains


def main():
    assert(len(sys.argv) == 2)  # Print usage error instead to be user friendly
    config_file = m.abspath2(sys.argv[1])

    print 'config %s' % config_file
    config = m.load_config(config_file)

    input_path = config[0]
    chains = get_chains(input_path)
    print 'inputs chains:'
    print chains

    for mc_chain_name in chains:
        m.run_experiment(config, mc_chain_name)
    print 'done'

if __name__ == '__main__':
    main()
