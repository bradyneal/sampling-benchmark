# Ryan Turner (turnerry@iro.umontreal.ca)
import ConfigParser
import os
import numpy as np

# ============================================================================
# TODO move everything here to general util file


def abspath2(fname):
    return os.path.abspath(os.path.expanduser(fname))


def is_safe_name(name_str, sep_chars='_-.'):
    safe = name_str.translate(None, sep_chars).isalnum()
    return safe


def build_output_name(mc_chain_name, model_name, pkl_ext, sep='_'):
    output_name = ''.join((mc_chain_name, sep, model_name, pkl_ext))
    assert(is_safe_name(output_name))
    return output_name


def load_np(input_path, fname, ext):
    fname = os.path.join(input_path, fname + ext)
    print 'loading %s' % fname
    assert(os.path.isabs(fname))
    X = np.genfromtxt(fname, dtype=float, delimiter=',', skip_header=0,
                      loose=False, invalid_raise=True)
    return X


def chomp(ss, ext):
    L = len(ext)
    if ss[-L:] != ext:
        raise Exception('string %s with extension %s when %s was expected' %
                        (ss, ss[-L:], ext))
    return ss[:-L]

# ============================================================================


def get_chains(input_path, ext):
    chains = sorted(chomp(fname, ext) for fname in os.listdir(input_path)
                    if ext in fname)
    return chains


def load_config(config_file):
    config = ConfigParser.RawConfigParser()
    assert(os.path.isabs(config_file))
    config.read(config_file)

    D = {}
    D['input_path'] = abspath2(config.get('phase1', 'output_path'))
    D['output_path'] = abspath2(config.get('phase2', 'output_path'))

    D['csv_ext'] = config.get('common', 'csv_ext')
    D['pkl_ext'] = config.get('common', 'pkl_ext')

    D['train_frac'] = config.getfloat('phase2', 'train_frac')
    assert(0.0 <= D['train_frac'] and D['train_frac'] <= 1.0)

    D['rnade_scratch'] = abspath2(config.get('phase2', 'rnade_scratch_dir'))
    return D
