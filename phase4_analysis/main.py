# Ryan Turner (turnerry@iro.umontreal.ca)
import ConfigParser
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from diagnostics import STD_DIAGNOSTICS
from metrics import STD_METRICS
from metrics import build_target, eval_inc

DATA_EXT = '.csv'

# TODO write combo func here, also add option to warn if not already abspath
abspath2 = os.path.abspath


def load_config(config_file):
    config = ConfigParser.RawConfigParser()
    assert(os.path.isabs(config_file))
    config.read(config_file)

    input_path = abspath2(config.get('phase3', 'output_path'))
    output_path = abspath2(config.get('phase4', 'output_path'))
    n_grid = config.getint('phase3', 'n_grid')

    meta_ext = config.get('common', 'meta_ext')
    exact_name = config.get('common', 'exact_name')

    csv_ext = config.get('common', 'csv_ext')
    assert(csv_ext == DATA_EXT)  # For now just assert instead of pass

    return input_path, output_path, meta_ext, exact_name, n_grid


def load_meta(input_path, fname, meta_ext, n_grid):
    fname = os.path.join(input_path, fname) + meta_ext
    df = pd.read_csv(fname, header=0)
    idx = pd.Series(index=df['grid_idx'].values, data=df['sample'].values)
    assert(idx[0] == 0)
    # TODO do this part in phase 3
    idx = idx.reindex(index=xrange(n_grid)).fillna(method='ffill').values
    assert(idx.dtype.kind == 'i')
    assert(idx.shape == (n_grid,))
    assert(np.all(np.isfinite(idx)))
    return idx


def find_traces(input_path, exact_name, ext=DATA_EXT):
    # Would prob not cause problem to remove sort here, but general good
    # practice with os.listdir()
    files = sorted(os.listdir(input_path))
    # Could assert unique here if we wanted to be sure

    examples_to_use = set()  # Only if we have exact do we try example
    samplers_to_use = set()  # Can exluce exact from list of samplers
    file_lookup = {}
    for fname in files:
        if not fname.endswith(DATA_EXT):
            continue

        # TODO create general parser for filenames in the project util
        # also assert safe name
        # Warning! hash seems to have _ in it somestimes atm
        case_name, temp_str = fname.rsplit('-', 1)  # Chop off hash part+ext
        assert(len(temp_str) == 6 + len(ext))  # For now, TODO remove
        curr_example, curr_sampler = case_name.rsplit('_', 1)

        S = file_lookup.setdefault((curr_example, curr_sampler), set())
        S.add(fname)  # Use set to ensure don't use chain more than once

        if curr_sampler == exact_name:  # Only use example if have exact result
            examples_to_use.add(curr_example)
        else:
            samplers_to_use.add(curr_sampler)  # Don't add exact to list
    examples_to_use = sorted(examples_to_use)
    samplers_to_use = sorted(samplers_to_use)
    return samplers_to_use, examples_to_use, file_lookup


def load_chain(input_path, fname):
    # TODO move to common util, check abs path
    X = np.loadtxt(os.path.join(input_path, fname), delimiter=',')
    return X


def dump_results(df, output_path, tbl_name, ext=DATA_EXT):
    fname = os.path.join(output_path, tbl_name + ext)
    df.to_csv(fname, na_rep='', header=True, index=True)
    # Also return where it was written in case we want to log it
    return fname


def xr_mean_lte(D, thold, dim):
    # TODO anyway to keep float without +0?
    P = (D <= thold).sum(dim=dim) / ((~D.isnull()).sum(dim=dim) + 0.0)
    return P


def init_data_array(coords, value=np.nan):
    shape = [len(grid) for _, grid in coords]
    X = xr.DataArray(value + np.zeros(shape), coords=coords)
    return X


def main():
    assert(len(sys.argv) == 2)
    config_file = abspath2(sys.argv[1])

    # TODO move to dict rep
    input_path, output_path, meta_ext, exact_name, n_grid = \
        load_config(config_file)

    samplers, examples, file_lookup = find_traces(input_path, exact_name)
    print 'found %d samplers and %d examples' % (len(samplers), len(examples))
    print '%d files in lookup table' % \
        sum(len(file_lookup[k]) for k in file_lookup)
    # Might be a better rule, should almost be constant across entries
    n_chains = max(len(file_lookup[k]) for k in file_lookup)
    print 'appears to be %d chains per case' % n_chains

    # Will remove this later when the list gets large
    print samplers
    print examples

    # Setup diagnostic datastruct
    # TODO eventually switch this to xr to for consistency
    cols = pd.MultiIndex.from_product([STD_DIAGNOSTICS.keys(), examples],
                                      names=['diagnostic', 'example'])
    diagnostic_df = pd.DataFrame(index=samplers, columns=cols, dtype=float)
    assert(np.all(diagnostic_df.isnull().values))  # init at nan

    # Setup perf datastruct, easy to swap order in xr
    metrics = STD_METRICS.keys()
    coords = [('time', xrange(n_grid)), ('metric', metrics),
              ('chain', xrange(n_chains)), ('sampler', samplers),
              ('example', examples)]
    perf = init_data_array(coords)
    perf_target = init_data_array([('metric', metrics), ('example', examples)])

    # Aggregate all the performance numbers into huge array
    for example in examples:
        fname, = file_lookup[(example, exact_name)]  # singleton set
        exact_chain = load_chain(input_path, fname)
        for metric in metrics:
            perf_target.loc[metric, example] = build_target(exact_chain, metric)
        for sampler in samplers:
            all_chains = []  # will be used by diags
            # Go in sorted order to keep it reproducible
            file_list = sorted(file_lookup[(example, sampler)])
            print 'found %d / %d chains for %s x %s' % \
                (len(file_list), n_chains, example, sampler)
            for c_num, fname in enumerate(file_list):
                curr_chain = load_chain(input_path, fname)
                all_chains.append(curr_chain)

                idx = load_meta(input_path, fname, meta_ext, n_grid)
                for metric in metrics:
                    perf_curr = eval_inc(exact_chain, curr_chain, metric, idx)
                    assert(perf_curr.shape == (n_grid,))
                    perf.loc[:, metric, c_num, sampler, example] = perf_curr
            for diag_name, diag_f in STD_DIAGNOSTICS.iteritems():
                score = diag_f(all_chains)
                diagnostic_df.loc[sampler, (diag_name, example)] = score

    # Now slice and dice to get summaries
    for example in examples:
        perf_slice = perf.sel(time=n_grid - 1, example=example)
        # Resulting are metric x sampler
        # TODO transpose
        mean_perf = perf_slice.mean(dim='chain', skipna=True)
        dump_results(mean_perf.to_pandas().T, output_path, 'perf-%s' % example)
        target_rate = xr_mean_lte(perf_slice,
                                  perf_target.sel(example=example),
                                  dim='chain')
        dump_results(target_rate.to_pandas().T, output_path,
                     'perf-rate-%s' % example)

    for metric in metrics:
        perf_slice = perf.sel(metric=metric)
        # Resulting are time x sampler
        time_perf = perf_slice.mean(dim=('chain', 'example'), skipna=True)
        dump_results(time_perf.to_pandas(), output_path,
                     'time-perf-%s' % metric)
        target_rate_v_time = \
            xr_mean_lte(perf_slice, perf_target.sel(metric=metric),
                        dim=('chain', 'example'))
        dump_results(target_rate_v_time.to_pandas(), output_path,
                     'time-perf-rate-%s' % metric)

    # Resulting is metric x sampler
    # TODO transpose
    final_perf = perf.isel(time=-1).mean(dim=('chain', 'example'), skipna=True)
    dump_results(final_perf.to_pandas().T, output_path, 'final-perf')
    final_perf_rate = xr_mean_lte(perf.isel(time=-1), perf_target,
                                  dim=('chain', 'example'))
    dump_results(final_perf_rate.to_pandas().T, output_path, 'final-perf-rate')

    dump_results(diagnostic_df, output_path, 'diagnostic')

    # Could also include option to dump everything in netCDF if we want
    print 'done'

if __name__ == '__main__':
    main()
