# Ryan Turner (turnerry@iro.umontreal.ca)
import ConfigParser
import os
from tempfile import NamedTemporaryFile

# ============================================================================
# TODO move everything here to general util file


def abspath2(fname):
    return os.path.abspath(os.path.expanduser(fname))


def is_safe_name(name_str, sep_chars='_-'):
    safe = name_str.translate(None, sep_chars).isalnum()
    return safe


def get_temp_filename(dir_, prefix, ext):
    # Warning! NamedTemporaryFile sometimes put _ in the random string, we need
    # to be careful this does not mess up our convention. There does not appear
    # to be a way to stop this.
    tf = NamedTemporaryFile(suffix=ext, prefix=prefix, dir=dir_, delete=False)
    fname = tf.name
    tf.close()
    assert(os.path.isabs(fname))
    return fname


def chomp(ss, ext):
    L = len(ext)
    assert(ss[-L:] == ext)
    return ss[:-L]

# ============================================================================


def get_model_list(input_path, ext):
    L = sorted(chomp(fname, ext) for fname in os.listdir(input_path))
    return L


def build_output_name(param_name, sampler, sep='_', sub_sep='-'):
    output_name = ''.join((param_name, sep, sampler, sub_sep))
    assert(is_safe_name(output_name))
    return output_name


def load_config(config_file):
    config = ConfigParser.RawConfigParser()
    assert(os.path.isabs(config_file))
    config.read(config_file)

    D = {}
    D['input_path'] = abspath2(config.get('phase2', 'output_path'))
    D['output_path'] = abspath2(config.get('phase3', 'output_path'))

    D['t_grid_ms'] = config.getint('phase3', 'time_grid_ms')
    assert(D['t_grid_ms'] > 0)
    D['n_grid'] = config.getint('phase3', 'n_grid')
    assert(D['n_grid'] > 0)
    D['n_exact'] = config.getint('phase3', 'n_exact')
    assert(D['n_exact'] > 0)
    D['n_chains'] = config.getint('phase3', 'n_chains')
    assert(D['n_chains'] >= 0)

    D['csv_ext'] = config.get('common', 'csv_ext')
    D['pkl_ext'] = config.get('common', 'pkl_ext')
    D['meta_ext'] = config.get('common', 'meta_ext')
    D['exact_name'] = config.get('common', 'exact_name')
    assert(D['exact_name'].isalnum())

    return D
