# Ryan Turner (turnerry@iro.umontreal.ca)
import ConfigParser
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
from diagnostics import STD_DIAGNOSTICS
import fileio as io
from metrics import STD_METRICS
from metrics import build_target, eval_inc

SAMPLE_INDEX_COL = 'sample'
DATA_EXT = '.csv'

# ============================================================================
# TODO move everything here to general util file


def init_data_array(coords, value=np.nan):
    shape = [len(grid) for _, grid in coords]
    X = xr.DataArray(value + np.zeros(shape), coords=coords)
    return X


def xr_mean_lte(D, thold, dim):
    # Will phase this out when we switch to n_eff
    P = (D <= thold).sum(dim=dim) / ((~D.isnull()).sum(dim=dim) + 0.0)
    return P

# ============================================================================


def load_config(config_file):
    config = ConfigParser.RawConfigParser()
    assert(os.path.isabs(config_file))
    config.read(config_file)

    input_path = io.abspath2(config.get('phase3', 'output_path'))
    output_path = io.abspath2(config.get('phase4', 'output_path'))
    n_grid = config.getint('phase3', 'n_grid')

    meta_ext = config.get('common', 'meta_ext')
    exact_name = config.get('common', 'exact_name')

    csv_ext = config.get('common', 'csv_ext')
    assert(csv_ext == DATA_EXT)  # For now just assert instead of pass

    return input_path, output_path, meta_ext, exact_name, n_grid


def load_meta(input_path, fname, meta_ext, n_grid):
    fname = os.path.join(input_path, fname) + meta_ext
    df = pd.read_csv(fname, header=0)
    sample_idx = df[SAMPLE_INDEX_COL].values
    # Do some validation before returning
    assert(sample_idx.shape == (n_grid,))
    assert(sample_idx.dtype.kind == 'i')
    assert(sample_idx[0] == 0)
    assert(np.all(np.isfinite(sample_idx)))
    assert(np.all(np.diff(sample_idx) >= 0))
    return sample_idx


def main():
    assert(len(sys.argv) == 2)
    config_file = io.abspath2(sys.argv[1])

    # TODO move to dict rep
    input_path, output_path, meta_ext, exact_name, n_grid = \
        load_config(config_file)

    samplers, examples, file_lookup = \
        io.find_traces(input_path, exact_name, DATA_EXT)
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

    # TODO use effective n as metric, can also use normalized RMSE for traditional
    # Units need to make sense in all calculation, consider robust scaler based on exact

    # Aggregate all the performance numbers into huge array
    for example in examples:
        fname, = file_lookup[(example, exact_name)]  # singleton set
        exact_chain = io.load_np(input_path, fname, ext='')
        for metric in metrics:
            perf_target.loc[metric, example] = build_target(exact_chain, metric)
        for sampler in samplers:
            all_chains = []  # will be used by diags
            # Go in sorted order to keep it reproducible
            file_list = sorted(file_lookup[(example, sampler)])
            print 'found %d / %d chains for %s x %s' % \
                (len(file_list), n_chains, example, sampler)
            for c_num, fname in enumerate(file_list):
                curr_chain = io.load_np(input_path, fname, ext='')
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
        io.save_pd(mean_perf.to_pandas().T, output_path,
                   'perf-%s' % example, DATA_EXT)
        target_rate = xr_mean_lte(perf_slice,
                                  perf_target.sel(example=example),
                                  dim='chain')
        io.save_pd(target_rate.to_pandas().T, output_path,
                   'perf-rate-%s' % example, DATA_EXT)

    for metric in metrics:
        perf_slice = perf.sel(metric=metric)
        # Resulting are time x sampler
        time_perf = perf_slice.mean(dim=('chain', 'example'), skipna=True)
        io.save_pd(time_perf.to_pandas(), output_path,
                   'time-perf-%s' % metric, DATA_EXT)
        target_rate_v_time = \
            xr_mean_lte(perf_slice, perf_target.sel(metric=metric),
                        dim=('chain', 'example'))
        io.save_pd(target_rate_v_time.to_pandas(), output_path,
                   'time-perf-rate-%s' % metric, DATA_EXT)

    # Resulting is metric x sampler
    # TODO transpose
    final_perf = perf.isel(time=-1).mean(dim=('chain', 'example'), skipna=True)
    io.save_pd(final_perf.to_pandas().T, output_path, 'final-perf', DATA_EXT)
    final_perf_rate = xr_mean_lte(perf.isel(time=-1), perf_target,
                                  dim=('chain', 'example'))
    io.save_pd(final_perf_rate.to_pandas().T, output_path,
               'final-perf-rate', DATA_EXT)

    io.save_pd(diagnostic_df, output_path, 'diagnostic', DATA_EXT)

    # Could also include option to dump everything in netCDF if we want
    print 'done'

if __name__ == '__main__':
    main()
