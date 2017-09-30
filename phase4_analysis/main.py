# Ryan Turner (turnerry@iro.umontreal.ca)
import ConfigParser
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xarray as xr
from diagnostics import STD_DIAGNOSTICS, ESS
import fileio as io
from metrics import STD_METRICS, STD_METRICS_REF
from metrics import eval_inc

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


def xr_hmean(X, dim=None, skipna=SKIPNA):
    '''Note this departs from xarray default for skipna.'''
    hm = 1.0 / ((1.0 / X).mean(dim=dim, skipna=skipna))
    return hm


def xr_to_df_hier(da, index, column, sub_column):
    '''This could be done generally with xr.to_dataframe() and then unstack.'''
    D = {}
    for col_val in da.coords[column].values:
        df = da.loc[{column: col_val}].to_pandas()
        # xr.to_pandas() doesn't let us specify the way we want it:
        if df.index.name != index:
            df = df.T
        assert(df.index.name == index)
        assert(df.columns.name == sub_column)
        D[col_val] = df
    df = pd.concat(D, axis=1)
    assert(df.columns.names == [None, sub_column])
    df.columns.names = [column, sub_column]
    assert(df.index.name == index)
    assert(df.columns.names == [column, sub_column])
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


def save_metric_summary(err_xr, n_count_xr, ess_ref_xr, output_path, ext):
    examples = err_xr.coords['example'].values
    samplers = err_xr.coords['sampler'].values
    metrics = err_xr.coords['metric'].values

    D = {'err': err_xr}
    # Note: coordinate ordering may change
    D['ess'] = ess_ref_xr / D['err']
    D['eff'] = D['ess'] / n_count_xr

    for space, X in D.iteritems():
        for ex in examples:
            df = xr_to_df_hier(X.sel(example=ex), 'time', 'sampler', 'metric')
            io.save_pd(df, output_path, 'ex-%s-%s' % (ex, space), ext)
        for sam in samplers:
            df = xr_to_df_hier(X.sel(sampler=sam), 'time', 'example', 'metric')
            io.save_pd(df, output_path, 'sam-%s-%s' % (sam, space), ext)
        for met in metrics:
            df = xr_to_df_hier(X.sel(metric=met), 'time', 'sampler', 'example')
            io.save_pd(df, output_path, 'met-%s-%s' % (met, space), ext)

    # Some averaging for a final summary score
    hm_n_count = xr_hmean(n_count_xr.isel(time=-1), dim='example')
    final_perf = err_xr.isel(time=-1).mean(dim='example', skipna=SKIPNA)
    df = final_perf.to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'metric')
    io.save_pd(df, output_path, 'final-%s' % 'err', ext)
    ess = (ess_ref_xr / final_perf).T
    df = ess.to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'metric')
    io.save_pd(df, output_path, 'final-%s' % 'ess', ext)
    df = (ess / hm_n_count).to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'metric')
    io.save_pd(df, output_path, 'final-%s' % 'eff', ext)


def build_metrics_array(samplers, examples, metrics, file_lookup, config,
                        bootstrap_test=False):
    '''Aggregate all the performance numbers into huge array'''
    n_chains, n_grid = config['n_chains'], config['n_grid']
    print 'expect %d chains per case' % n_chains

    coords = [('time', xrange(n_grid)), ('sampler', samplers),
              ('example', examples), ('metric', metrics)]
    perf = init_data_array(coords)
    # Skip metric for n_count
    n_count = init_data_array(coords[:-1])

    # Final perf with synched scores, skip time. Eventually we want to be more
    # consistent on using either pandas or xarray once we know what is more
    # appropriate.
    assert(set(metrics).isdisjoint(STD_DIAGNOSTICS.keys()))
    metrics_sync = STD_DIAGNOSTICS.keys() + metrics
    cols = pd.MultiIndex.from_product([metrics_sync, examples],
                                      names=['metric', 'example'])
    perf_sync = pd.DataFrame(index=samplers, columns=cols, dtype=float)
    perf_sync.index.name = 'sampler'
    assert(np.all(perf_sync.isnull().values))  # init at nan
    n_count_sync = pd.DataFrame(index=samplers, columns=examples, dtype=float)
    n_count_sync.index.name = 'sampler'
    assert(np.all(n_count_sync.isnull().values))  # init at nan

    for example in examples:
        fname, = file_lookup[(example, config['exact_name'])]  # singleton set
        exact_chain = io.load_np(config['input_path'], fname, ext='')
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
                all_chains.append(curr_chain)

            # Do analyses that can be done with unequal length chains
            print 'async analysis'
            for metric in metrics:
                err = eval_inc(exact_chain, all_chains, metric, all_meta)
                assert(err.shape == (n_grid,))
                perf.loc[:, sampler, example, metric] = err
            n_count.loc[:, sampler, example] = hmean(all_meta, axis=1)

            print 'sync analysis'
            # Do analyses that can only be done @ end with equal len chains
            all_chains = combine_chains(all_chains)  # Now np array
            min_n = all_chains.shape[1]
            final_idx = min_n + np.zeros((1, all_chains.shape[0]), dtype=int)
            n_count_sync.loc[sampler, example] = min_n
            for metric in metrics:
                err, = eval_inc(exact_chain, all_chains, metric, final_idx)
                assert(np.ndim(err) == 0)
                perf_sync.loc[sampler, (metric, example)] = err
            print 'diagnostics'
            for diag_name, diag_f in STD_DIAGNOSTICS.iteritems():
                print diag_name
                score = diag_f(all_chains)
                assert(np.ndim(score) == 0)
                perf_sync.loc[sampler, (diag_name, example)] = score
    return perf, n_count, perf_sync, n_count_sync


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

    # This could get big
    print samplers
    print examples

    metrics = STD_METRICS.keys()
    R = build_metrics_array(samplers, examples, metrics, file_lookup, config)
    perf, n_count, perf_sync, n_count_sync = R

    ess_ref = xr.DataArray(data=STD_METRICS_REF.values(),
                           coords=[('metric', STD_METRICS_REF.keys())])
    print 'using reference values'
    print ess_ref.to_pandas().to_string()

    # Save metrics
    save_metric_summary(perf, n_count, ess_ref, config['output_path'], ext)

    # Save diagnostics
    io.save_pd(perf_sync, config['output_path'], 'perf_sync', ext)

    # Report on ESS measures
    D = {ESS: perf_sync[ESS]}
    for metric in metrics:
        D[metric] = STD_METRICS_REF[metric] / perf_sync[metric]
        assert(D[metric].index.name == 'sampler')
        assert(D[metric].columns.name == 'example')
    df = pd.concat(D, axis=1)
    assert(df.index.name == 'sampler')
    assert(df.columns.names == [None, 'example'])
    df.columns.names = ['metric', 'example']
    io.save_pd(df, config['output_path'], 'ess', ext)

    # Could also include option to dump everything in netCDF if we want
    print 'done'
    return perf, n_count

if __name__ == '__main__':
    perf, n_count = main()
