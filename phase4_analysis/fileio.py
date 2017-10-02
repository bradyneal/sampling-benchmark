import os
import numpy as np

TEMP_STR_LEN = 6

# TODO some of this should go to general util


def parse_sampler_name(fname, ext, sep='_', sub_sep='-'):
    # Warning! temp str suffix seems to have _ in it somestimes atm
    case_name, temp_str = fname.rsplit(sub_sep, 1)  # Chop off hash part+ext
    assert(len(temp_str) == TEMP_STR_LEN + len(ext))
    curr_example, curr_sampler = case_name.rsplit(sep, 1)
    return curr_example, curr_sampler


def abspath2(fname):
    return os.path.abspath(os.path.expanduser(fname))


def load_np(input_path, fname, ext):
    fname = os.path.join(input_path, fname + ext)
    print 'loading %s' % fname
    assert(os.path.isabs(fname))
    X = np.genfromtxt(fname, dtype=float, delimiter=',', skip_header=0,
                      loose=False, invalid_raise=True)
    return X


def save_pd(df, output_path, tbl_name, ext, index=True):
    fname = os.path.join(output_path, tbl_name + ext)
    print 'saving %s' % fname
    assert(os.path.isabs(fname))
    df.to_csv(fname, na_rep='', header=True, index=index)
    return fname


def find_traces(input_path, exact_name, ext, sep='_', sub_sep='-'):
    # Sort not needed here, but general good practice with os.listdir()
    files = sorted(os.listdir(input_path))
    # Could assert unique here if we wanted to be sure

    examples_to_use = set()  # Only if we have exact do we try example
    samplers_to_use = set()  # Can exluce exact from list of samplers
    file_lookup = {}
    for fname in files:
        if not fname.endswith(ext):
            continue  # skip .meta files
        curr_example, curr_sampler = \
            parse_sampler_name(fname, ext=ext, sep=sep, sub_sep=sub_sep)

        # Note: may contain examples and samplers not in the to_use lists
        S = file_lookup.setdefault((curr_example, curr_sampler), set())
        S.add(fname)  # Use set to ensure don't use chain more than once

        if curr_sampler == exact_name:  # Only use example if have exact result
            examples_to_use.add(curr_example)
        else:
            samplers_to_use.add(curr_sampler)  # Don't add exact to list
    examples_to_use = sorted(examples_to_use)
    samplers_to_use = sorted(samplers_to_use)
    return samplers_to_use, examples_to_use, file_lookup
