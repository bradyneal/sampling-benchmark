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
from metrics import STD_METRICS, STD_METRICS_REF
from metrics import eval_inc

SAMPLE_INDEX_COL = 'sample'


def init_data_array(coords, value=np.nan):
    '''Why is this not built into to xarray?'''
    shape = [len(grid) for _, grid in coords]
    X = xr.DataArray(value + np.zeros(shape), coords=coords)
    return X


def resample(X, N):
    idx = np.random.choice(X.shape[0], size=N, replace=True)
    XS = X[idx, :]
    return XS


def hmean(X, axis=None):
    '''Because the scipy gives an error on zero.'''
    hm = 1.0 / np.mean(1.0 / X, axis=axis)
    return hm


def xr_hmean(X, dim=None, skipna=False):
    '''Note this departs from xarray default for skipna.'''
    hm = 1.0 / ((1.0 / X).mean(dim=dim, skipna=skipna))
    return hm


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


def save_metric_summary(perf, n_count, ess_ref, output_path, ext):
    examples = perf.coords['example'].values
    metrics = perf.coords['metric'].values

    hm_n_count = xr_hmean(n_count, dim='example')

    for example in examples:
        ex_perf = perf.isel(time=-1).sel(example=example)
        df = ex_perf.to_pandas()
        assert(df.index.name == 'sampler')
        assert(df.columns.name == 'metric')
        io.save_pd(df, output_path, '%s-%s' % (example, 'err'), ext)
        # ess_ref is of dim (metrics,) => metric becomes first dim => need .T
        ess = (ess_ref / ex_perf).T
        df = ess.to_pandas()
        assert(df.index.name == 'sampler')
        assert(df.columns.name == 'metric')
        io.save_pd(df, output_path, '%s-%s' % (example, 'ess'), ext)
        df = (ess / n_count.isel(time=-1).sel(example=example)).to_pandas()
        assert(df.index.name == 'sampler')
        assert(df.columns.name == 'metric')
        io.save_pd(df, output_path, '%s-%s' % (example, 'eff'), ext)

    for metric in metrics:
        perf_slice = perf.sel(metric=metric)
        time_perf = perf_slice.mean(dim='example', skipna=False)
        df = time_perf.to_pandas()
        assert(df.index.name == 'time')
        assert(df.columns.name == 'sampler')
        io.save_pd(df, output_path, 'time-%s-%s' % (metric, 'err'), ext)
        ess = ess_ref.sel(metric=metric) / time_perf
        df = ess.to_pandas()
        assert(df.index.name == 'time')
        assert(df.columns.name == 'sampler')
        io.save_pd(df, output_path, 'time-%s-%s' % (metric, 'ess'), ext)
        df = (ess / hm_n_count).to_pandas()
        assert(df.index.name == 'time')
        assert(df.columns.name == 'sampler')
        io.save_pd(df, output_path, 'time-%s-%s' % (metric, 'eff'), ext)

    final_perf = perf.isel(time=-1).mean(dim='example', skipna=False)
    df = final_perf.to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'metric')
    io.save_pd(df, output_path, 'final-%s' % 'err', ext)
    ess = (ess_ref / final_perf).T
    df = ess.to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'metric')
    io.save_pd(df, output_path, 'final-%s' % 'ess', ext)
    df = (ess / hm_n_count.isel(time=-1)).to_pandas()
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
    # Final perf with synched scores, skip time
    perf_sync_final = init_data_array(coords[1:])

    # Setup diagnostic datastruct
    # TODO eventually switch this to xr to for consistency
    # TODO take diagnostic names as inputs for consistency??
    cols = pd.MultiIndex.from_product([STD_DIAGNOSTICS.keys(), examples],
                                      names=['diagnostic', 'example'])
    diag_df = pd.DataFrame(index=samplers, columns=cols, dtype=float)
    assert(np.all(diag_df.isnull().values))  # init at nan

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
            for metric in metrics:
                err = eval_inc(exact_chain, all_chains, metric, all_meta)
                assert(err.shape == (n_grid,))
                perf.loc[:, sampler, example, metric] = err
            n_count.loc[:, sampler, example] = hmean(all_meta, axis=1)

            # Do analyses that can only be done @ end with equal len chains
            all_chains = combine_chains(all_chains)  # Now np array
            min_n = all_chains.shape[1]
            final_idx = min_n + np.zeros((1, all_chains.shape[0]), dtype=int)
            for metric in metrics:
                err, = eval_inc(exact_chain, all_chains, metric, final_idx)
                assert(np.ndim(err) == 0)
                perf_sync_final.loc[sampler, example, metric] = err
            for diag_name, diag_f in STD_DIAGNOSTICS.iteritems():
                score = diag_f(all_chains)
                diag_df.loc[sampler, (diag_name, example)] = score
    return perf, n_count, perf_sync_final, diag_df


def main():
    assert(len(sys.argv) == 2)
    config_file = io.abspath2(sys.argv[1])

    config = load_config(config_file)

    samplers, examples, file_lookup = \
        io.find_traces(config['input_path'], config['exact_name'],
                       config['csv_ext'])
    print 'found %d samplers and %d examples' % (len(samplers), len(examples))
    print '%d files in lookup table' % \
        sum(len(file_lookup[k]) for k in file_lookup)

    # This could get big
    print samplers
    print examples

    metrics = STD_METRICS.keys()
    R = build_metrics_array(samplers, examples, metrics, file_lookup, config)
    perf, n_count, perf_sync_final, diag_df = R

    ess_ref = xr.DataArray(data=STD_METRICS_REF.values(),
                           coords=[('metric', STD_METRICS_REF.keys())])
    print 'using reference values'
    print ess_ref.to_pandas().to_string()

    # Save metrics
    save_metric_summary(perf, n_count, ess_ref,
                        config['output_path'], config['csv_ext'])

    # Save diagnostics
    io.save_pd(diag_df, config['output_path'], 'diagnostic', config['csv_ext'])

    # Just consider mean sync'd ESS now
    ess = ess_ref.sel(metric='mean') / perf_sync_final.sel(metric='mean')
    df = ess.to_pandas()
    assert(df.index.name == 'sampler')
    assert(df.columns.name == 'example')
    io.save_pd(df, config['output_path'], 'ess', config['csv_ext'])

    # Could also include option to dump everything in netCDF if we want
    print 'done'
    return perf, n_count

if __name__ == '__main__':
    perf, n_count = main()
