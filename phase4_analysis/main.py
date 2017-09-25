# Ryan Turner (turnerry@iro.umontreal.ca)
import ConfigParser
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xarray as xr
from diagnostics import STD_DIAGNOSTICS
import fileio as io
from metrics import STD_METRICS
from metrics import eval_inc

SAMPLE_INDEX_COL = 'sample'
DATA_EXT = '.csv'


def init_data_array(coords, value=np.nan):
    '''Why is this not built into to xarray?'''
    shape = [len(grid) for _, grid in coords]
    X = xr.DataArray(value + np.zeros(shape), coords=coords)
    return X


def combine_chains(chains):
    assert(len(chains) >= 1)
    D = chains[0].shape[1]
    assert(all(np.ndim(x) == 2 and x.shape[1] == D for x in chains))
    min_n = min(x.shape[0] for x in chains)
    assert(min_n > 0)

    # Take end since these are the best samples, could also thin to size
    combo = np.stack([x[-min_n:, :] for x in chains], axis=0)
    assert(combo.shape == (len(chains), min_n, D))
    return combo


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


def save_metric_summary(perf, output_path):
    # TODO lookup best skipna policy
    space = perf.coords['space'].values
    assert(np.ndim(space) == 0)  # Already sliced
    examples = perf.coords['example'].values
    metrics = perf.coords['metric'].values
    n_grid = len(perf.coords['time'])

    for example in examples:
        # We could make this tighter with time=-1 in sep sel() call
        perf_slice = perf.sel(time=n_grid - 1, example=example)
        mean_perf = perf_slice.mean(dim='chain', skipna=True)
        df = mean_perf.to_pandas()
        assert(df.index.name == 'sampler')
        assert(df.columns.name == 'metric')
        io.save_pd(df, output_path, '%s-%s' % (example, space), DATA_EXT)

    for metric in metrics:
        perf_slice = perf.sel(metric=metric)
        time_perf = perf_slice.mean(dim=('chain', 'example'), skipna=True)
        df = time_perf.to_pandas()
        assert(df.index.name == 'time')
        assert(df.columns.name == 'sampler')
        io.save_pd(df, output_path, 'time-%s-%s' % (metric, space), DATA_EXT)

    final_perf = perf.isel(time=-1).mean(dim=('chain', 'example'), skipna=True)
    df = final_perf.to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'metric')
    io.save_pd(df, output_path, 'final-%s' % space, DATA_EXT)


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
    # TODO lookup in config file
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

    spaces = ['err', 'ess', 'eff']
    metrics = STD_METRICS.keys()

    # Setup perf datastruct, easy to swap order in xr
    coords = [('time', xrange(n_grid)), ('chain', xrange(n_chains)),
              ('sampler', samplers), ('example', examples),
              ('metric', metrics), ('space', spaces)]
    perf = init_data_array(coords)
    # Final perf with synched scores
    coords = [('sampler', samplers), ('example', examples),
              ('metric', metrics), ('space', spaces)]
    perf_sync_final = init_data_array(coords)

    # Aggregate all the performance numbers into huge array
    # TODO pull out into function
    for example in examples:
        fname, = file_lookup[(example, exact_name)]  # singleton set
        exact_chain = io.load_np(input_path, fname, ext='')
        # Put all variables on same scale using the exact chain here, we can
        # change this to robust scaler if it gives us trouble with outliers.
        scaler = StandardScaler()
        exact_chain = scaler.fit_transform(exact_chain)
        for sampler in samplers:
            all_chains = []  # will be used by diags
            # Go in sorted order to keep it reproducible
            file_list = sorted(file_lookup[(example, sampler)])
            print 'found %d / %d chains for %s x %s' % \
                (len(file_list), n_chains, example, sampler)
            for c_num, fname in enumerate(file_list):
                curr_chain = io.load_np(input_path, fname, ext='')
                curr_chain = scaler.transform(curr_chain)
                all_chains.append(curr_chain)

                idx = load_meta(input_path, fname, meta_ext, n_grid)
                for metric in metrics:
                    R = eval_inc(exact_chain, curr_chain, metric, idx)
                    assert(all(err.shape == (n_grid,) for err in R))

                    # Save all 3 views on how to scale the error.
                    perf.loc[:, c_num, sampler, example, metric, 'err'] = R[0]
                    perf.loc[:, c_num, sampler, example, metric, 'ess'] = R[1]
                    perf.loc[:, c_num, sampler, example, metric, 'eff'] = R[2]

            # Now do analyses that can only be done @ end with multiple chains
            all_chains = combine_chains(all_chains)  # Now np array
            for metric, metric_f in STD_METRICS.iteritems():
                R = np.mean([metric_f(exact_chain, c) for c in all_chains],
                            axis=0)

                # Save all 3 views on how to scale the error.
                perf_sync_final.loc[sampler, example, metric, 'err'] = R[0]
                perf_sync_final.loc[sampler, example, metric, 'ess'] = R[1]
                perf_sync_final.loc[sampler, example, metric, 'eff'] = R[2]
            for diag_name, diag_f in STD_DIAGNOSTICS.iteritems():
                score = diag_f(all_chains)
                diagnostic_df.loc[sampler, (diag_name, example)] = score

    # Save metrics in all spaces
    for space in spaces:
        save_metric_summary(perf.sel(space=space), output_path)
    # Save diagnostics
    io.save_pd(diagnostic_df, output_path, 'diagnostic', DATA_EXT)

    # TODO clean up, this is just to start
    df = perf_sync_final.sel(metric='mean', space='ess').to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'example')
    io.save_pd(df, output_path, 'ess', DATA_EXT)

    # Could also include option to dump everything in netCDF if we want
    print 'done'
    return perf

if __name__ == '__main__':
    perf = main()
