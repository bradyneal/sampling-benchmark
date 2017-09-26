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

    D = {}
    D['input_path'] = io.abspath2(config.get('phase3', 'output_path'))
    D['output_path'] = io.abspath2(config.get('phase4', 'output_path'))

    D['n_grid'] = config.getint('phase3', 'n_grid')
    D['n_chains'] = config.getint('phase3', 'n_chains')

    D['csv_ext'] = config.get('common', 'csv_ext')
    D['meta_ext'] = config.get('common', 'meta_ext')
    D['exact_name'] = config.get('common', 'exact_name')

    return D


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


def save_metric_summary(perf, output_path, ext):
    # TODO lookup best skipna policy
    space = perf.coords['space'].values
    assert(np.ndim(space) == 0)  # Already sliced
    examples = perf.coords['example'].values
    metrics = perf.coords['metric'].values
    n_grid = len(perf.coords['time'])

    for example in examples:
        # We could make this tighter with time=-1 in sep sel() call
        df = perf.sel(time=n_grid - 1, example=example).to_pandas()
        assert(df.index.name == 'sampler')
        assert(df.columns.name == 'metric')
        io.save_pd(df, output_path, '%s-%s' % (example, space), ext)

    for metric in metrics:
        perf_slice = perf.sel(metric=metric)
        time_perf = perf_slice.mean(dim='example', skipna=True)
        df = time_perf.to_pandas()
        assert(df.index.name == 'time')
        assert(df.columns.name == 'sampler')
        io.save_pd(df, output_path, 'time-%s-%s' % (metric, space), ext)

    final_perf = perf.isel(time=-1).mean(dim='example', skipna=True)
    df = final_perf.to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'metric')
    io.save_pd(df, output_path, 'final-%s' % space, ext)


def main():
    assert(len(sys.argv) == 2)
    config_file = io.abspath2(sys.argv[1])

    config = load_config(config_file)

    n_chains, n_grid = config['n_chains'], config['n_grid']
    print 'expect %d chains per case' % n_chains

    samplers, examples, file_lookup = \
        io.find_traces(config['input_path'], config['exact_name'],
                       config['csv_ext'])
    print 'found %d samplers and %d examples' % (len(samplers), len(examples))
    print '%d files in lookup table' % \
        sum(len(file_lookup[k]) for k in file_lookup)

    # This could get big
    print samplers
    print examples

    # Setup diagnostic datastruct
    # TODO eventually switch this to xr to for consistency
    cols = pd.MultiIndex.from_product([STD_DIAGNOSTICS.keys(), examples],
                                      names=['diagnostic', 'example'])
    diag_df = pd.DataFrame(index=samplers, columns=cols, dtype=float)
    assert(np.all(diag_df.isnull().values))  # init at nan

    spaces = ['err', 'ess', 'eff']
    metrics = STD_METRICS.keys()

    # Setup perf datastruct, easy to swap order in xr
    coords = [('time', xrange(n_grid)),
              ('sampler', samplers), ('example', examples),
              ('metric', metrics), ('space', spaces)]
    perf = init_data_array(coords)
    # Final perf with synched scores, skip time
    perf_sync_final = init_data_array(coords[1:])

    # Aggregate all the performance numbers into huge array
    # TODO pull out into function
    for example in examples:
        fname, = file_lookup[(example, config['exact_name'])]  # singleton set
        exact_chain = io.load_np(config['input_path'], fname, ext='')
        # Put all variables on same scale using the exact chain here, we can
        # change this to robust scaler if it gives us trouble with outliers.
        scaler = StandardScaler()
        exact_chain = scaler.fit_transform(exact_chain)
        for sampler in samplers:
            # Go in sorted order to keep it reproducible
            file_list = sorted(file_lookup[(example, sampler)])
            print 'found %d / %d chains for %s x %s' % \
                (len(file_list), n_chains, example, sampler)

            all_chains = []
            all_meta = np.zeros((n_grid, len(file_list)), dtype=int)
            for ii, fname in enumerate(file_list):
                curr_chain = io.load_np(config['input_path'], fname, ext='')
                all_chains.append(scaler.transform(curr_chain))
                idx = load_meta(config['input_path'],
                                fname, config['meta_ext'], n_grid)
                all_meta[:, ii] = idx

            # Do analyses that can be done with unequal length chains
            for metric in metrics:
                R = eval_inc(exact_chain, all_chains, metric, all_meta)
                assert(all(err.shape == (n_grid,) for err in R))

                # Save all 3 views on how to scale the error.
                perf.loc[:, sampler, example, metric, 'err'] = R[0]
                perf.loc[:, sampler, example, metric, 'ess'] = R[1]
                perf.loc[:, sampler, example, metric, 'eff'] = R[2]

            # Do analyses that can only be done @ end with equal len chains
            all_chains = combine_chains(all_chains)  # Now np array
            for metric, metric_f in STD_METRICS.iteritems():
                # TODO fix this, this is garbage results for new metric_f
                R = np.mean([metric_f(exact_chain, c) for c in all_chains],
                            axis=0)

                # Save all 3 views on how to scale the error.
                perf_sync_final.loc[sampler, example, metric, 'err'] = R[0]
                perf_sync_final.loc[sampler, example, metric, 'ess'] = R[1]
                perf_sync_final.loc[sampler, example, metric, 'eff'] = R[2]
            for diag_name, diag_f in STD_DIAGNOSTICS.iteritems():
                score = diag_f(all_chains)
                diag_df.loc[sampler, (diag_name, example)] = score

    # Save metrics in all spaces
    for space in spaces:
        save_metric_summary(perf.sel(space=space), config['output_path'],
                            config['csv_ext'])
    # Save diagnostics
    io.save_pd(diag_df, config['output_path'], 'diagnostic', config['csv_ext'])

    # TODO clean up, this is just to start
    df = perf_sync_final.sel(metric='mean', space='ess').to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'example')
    io.save_pd(df, config['output_path'], 'ess', config['csv_ext'])

    # Could also include option to dump everything in netCDF if we want
    print 'done'
    return perf

if __name__ == '__main__':
    perf = main()
