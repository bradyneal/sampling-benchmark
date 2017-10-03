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
from metrics import MOMENT_METRICS, OTHER_METRICS
from metrics import eval_inc, eval_total, eval_pooled

SAMPLE_INDEX_COL = 'sample'
SKIPNA = True


def resample(X, N):
    idx = np.random.choice(X.shape[0], size=N, replace=True)
    XS = X[idx, :]
    return XS


def hmean(X, axis=None):
    '''Because the scipy gives an error on zero.'''
    hm = 1.0 / np.mean(1.0 / X, axis=axis)
    return hm


def init_data_array(coords, value=np.nan):
    '''Why is this not built into to xarray?'''
    shape = [len(grid) for _, grid in coords]
    X = xr.DataArray(value + np.zeros(shape), coords=coords)
    return X


def xr_unstack(da, index):
    dims_list = list(da.coords.dims)
    dims_list.remove(index)

    df = da.to_dataframe('foo').unstack(level=index)
    df.columns = df.columns.droplevel(level=0)
    df = df.T
    assert(df.index.name == index)
    assert(df.columns.names == dims_list)
    return df


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


def build_metrics_array(samplers, examples, metrics, file_lookup, config,
                        bootstrap_test=False):
    '''Aggregate all the performance numbers into huge array'''
    n_chains, n_grid = config['n_chains'], config['n_grid']
    print 'expect %d chains per case' % n_chains

    # Will want to settle on all xr or all pd once we know what is easier.
    coords = [('time', xrange(n_grid)), ('sampler', samplers),
              ('example', examples), ('metric', metrics)]
    perf = init_data_array(coords)
    # Skip metric for n_count
    n_count = init_data_array(coords[:-1])

    cols = pd.MultiIndex.from_product([samplers, examples, metrics + ['N']],
                                      names=['sampler', 'example', 'metric'])
    perf_df = pd.DataFrame(index=xrange(n_grid), columns=cols)
    perf_df.index.name = 'time'

    # Assume later that these keys are distinct
    assert(set(metrics).isdisjoint(STD_DIAGNOSTICS.keys()))
    metrics_sync = STD_DIAGNOSTICS.keys() + metrics
    sync_perf = {}
    for example in examples:
        fname, = file_lookup[(example, config['exact_name'])]  # singleton set
        exact_chain = io.load_np(config['input_path'], fname, ext='')
        D = exact_chain.shape[1]
        # In ESS calculations we assume that var=1, so we need standard scaler
        # and not robust, but maybe we could add warning if the two diverge.
        scaler = StandardScaler()
        exact_chain = scaler.fit_transform(exact_chain)
        for sampler in samplers:
            # Go in sorted order to keep it reproducible
            file_list = sorted(file_lookup.get((example, sampler), []))
            print 'found %d / %d chains for %s x %s' % \
                (len(file_list), n_chains, example, sampler)
            if len(file_list) == 0:  # Nothing to do
                continue

            # Iterate over chains into one big list data struct
            all_chains = []
            all_meta = np.zeros((n_grid, len(file_list)), dtype=int)
            for ii, fname in enumerate(file_list):
                all_meta[:, ii] = load_meta(config['input_path'], fname,
                                            config['meta_ext'], n_grid)
                if bootstrap_test:
                    curr_chain = resample(exact_chain, all_meta[-1, ii])
                else:  # Load actual data
                    curr_chain = io.load_np(config['input_path'], fname, '')
                    curr_chain = scaler.transform(curr_chain)
                assert(curr_chain.shape[1] == D)
                all_chains.append(curr_chain)

            # Do analyses that can be done with unequal length chains
            print 'async analysis'
            for metric in metrics:
                err = eval_inc(exact_chain, all_chains, metric, all_meta)
                assert(err.shape == (n_grid,))
                perf.loc[:, sampler, example, metric] = err
                perf_df[(sampler, example, metric)] = err
            n_count.loc[:, sampler, example] = hmean(all_meta, axis=1)
            perf_df[(sampler, example, 'N')] = hmean(all_meta, axis=1)

            print 'sync analysis'
            # Do analyses that can only be done @ end with equal len chains
            all_chains = combine_chains(all_chains)  # Now np array
            df = pd.DataFrame(index=xrange(D), columns=metrics_sync)
            df.index.name = 'dim'
            for metric in metrics:
                err = eval_total(exact_chain, all_chains, metric)
                assert(err.shape == (D,))
                df[metric] = err
                err = eval_pooled(exact_chain, all_chains, metric)
                assert(err.shape == (D,))
                df[metric + '_pooled'] = err
            for diag_name, diag_f in STD_DIAGNOSTICS.iteritems():
                score = diag_f(all_chains)
                assert(score.shape == (D,))
                df[diag_name] = score
            df['D'] = D
            df['N'] = all_chains.shape[1]
            df['n_chains'] = all_chains.shape[0]
            sync_perf[(sampler, example)] = df
    sync_perf = pd.concat(sync_perf, axis=0)
    assert(sync_perf.index.names == [None, None, 'dim'])
    sync_perf.index.names = ['sampler', 'example', 'dim']
    sync_perf.reset_index(drop=False, inplace=True)

    # Compare aggregation with pandas vs xarray
    perf_df2 = xr_unstack(perf, 'time')
    for metric in metrics:
        chk1 = perf_df.xs(metric, axis=1, level='metric')
        chk2 = perf_df2.xs(metric, axis=1, level='metric')
        assert(chk1.equals(chk2))
    chk1 = xr_unstack(n_count, 'time')
    chk2 = perf_df.xs('N', axis=1, level='metric')
    assert(chk1.equals(chk2))

    # We could add an extra example which is average of all examples, or can
    # do in next phase.
    return perf_df, sync_perf


def main():
    assert(len(sys.argv) == 2)
    config_file = io.abspath2(sys.argv[1])

    config = load_config(config_file)
    ext = config['csv_ext']

    samplers, examples, file_lookup = \
        io.find_traces(config['input_path'], config['exact_name'], ext)
    print 'found %d samplers and %d examples' % (len(samplers), len(examples))
    print '%d files in lookup table' % \
        sum(len(file_lookup[k]) for k in file_lookup)

    # examples = examples[:3]  # TODO remove

    # This could get big
    print samplers
    print examples

    metrics = MOMENT_METRICS.keys() + OTHER_METRICS.keys()
    R = build_metrics_array(samplers, examples, metrics, file_lookup, config)
    perf_df, sync_perf = R

    # Save TS, make sure it has enough info to compute ess and eff
    io.save_pd(perf_df, config['output_path'], 'perf', ext)

    # Save diagnostics, make sure it has enough info to compute ess and eff
    io.save_pd(sync_perf, config['output_path'], 'perf_sync', ext, index=False)

    # Could also include option to dump everything in netCDF if we want
    print 'done'

if __name__ == '__main__':
    main()
