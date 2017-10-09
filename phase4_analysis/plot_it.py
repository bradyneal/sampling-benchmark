# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from metrics import METRICS_REF

import bt.benchmark_tools_regr as btr
import bt.data_splitter as ds
from bt.sciprint import just_format_it, NAN_STR

import matplotlib.pyplot as plt

DIAGS = ('Geweke', 'ESS', 'Gelman_Rubin')  # TODO import from elsewhere

# TODO re-run with no clip in perf
#   also adjust MIN_ESS to per chain


def ESS_EB_fac(n_estimates, conf=0.95):
    P_ub = 0.5 * (1.0 + conf)
    P_lb = 1.0 - P_ub

    lb_fac = n_estimates / ss.chi2.ppf(P_ub, n_estimates)
    ub_fac = n_estimates / ss.chi2.ppf(P_lb, n_estimates)
    return lb_fac, ub_fac


def ESS_deviation(err, ess, n_chains, ref=1.0):
    C = (n_chains * ess) / ref
    dev = -ss.norm.ppf(ss.chi2.sf(C * err, n_chains))
    return dev


def aggregate_df(df, diag_agg=np.mean):
    gdf = df.groupby(['sampler', 'example'])

    aggf = {k: diag_agg for k in DIAGS}
    for metric in METRICS_REF.keys():
        aggf[metric] = np.mean
        aggf[metric + '_pooled'] = np.mean
    aggf['N'] = np.max
    aggf['D'] = np.max
    aggf['n_chains'] = np.max

    R = gdf.agg(aggf)
    # TODO assert size == D
    assert((gdf['N'].min() == R['N']).all())  # double check
    R.reset_index(drop=False, inplace=True)
    return R


def augment_df(df):
    # Note: this is happening inplace!
    n_ref = df.groupby('example')['N'].median()
    df['n_ref'] = df['example'].map(n_ref)

    df['ESS_pooled'] = df['ESS']
    df['ESS'] = df['ESS_pooled'] / df['n_chains']

    df['NESS_pooled'] = df['ESS_pooled'] / df['n_ref']
    df['NESS'] = df['ESS'] / df['n_ref']

    df['eff'] = df['ESS'] / df['N']
    df['eff_pooled'] = df['ESS_pooled'] / df['N']

    df['TG'] = df['Geweke'].abs()
    df['TGR'] = (df['Gelman_Rubin'] - 1.0).abs()

    for metric in sorted(METRICS_REF.keys()):
        metric_ref = METRICS_REF[metric]
        df['real_ess_' + metric] = metric_ref / df[metric]
        df['real_ness_' + metric] = df['real_ess_' + metric] / df['n_ref']
        df['real_eff_' + metric] = df['real_ess_' + metric] / df['N']
        df['real_essd_' + metric] = ESS_deviation(df[metric].values,
            df['ESS'].values, df['n_chains'].values, metric_ref)

        metric = metric + '_pooled'
        df['real_ess_' + metric] = metric_ref / df[metric]
        df['real_ness_' + metric] = df['real_ess_' + metric] / df['n_ref']
        total_samples = df['N'] * df['n_chains']
        df['real_eff_' + metric] = df['real_ess_' + metric] / total_samples
        df['real_essd_' + metric] = ESS_deviation(df[metric].values,
            df['ESS_pooled'].values, 1.0, metric_ref)
    return df  # return anyway


def plot_by(df, x, y, by, fac_lines=(1.0,)):
    # TODO make logscale arg
    plt.figure()
    gdf = df.groupby(by)
    for name, sdf in gdf:
        plt.loglog(sdf[x].values, sdf[y].values, '.', label=name, alpha=0.5)
    xgrid = np.logspace(np.log10(df[x].min()), np.log10(df[x].max()), 100)
    for f in fac_lines:
        plt.loglog(xgrid, f * xgrid, 'k--')
    plt.legend(loc=3, ncol=4, fontsize=6,
               bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    plt.grid()
    plt.xlabel(x)
    plt.ylabel(y)


def all_pos_finite(X):
    R = np.all((0.0 < X) & (X < np.inf))  # Will catch nan too
    return R


def all_finite(X):
    R = np.all(np.isfinite(X))  # Will catch nan too
    return R


def try_models(df_train, df_test, metric, feature_list, target, methods):
    loss_dict = btr.STD_REGR_LOSS

    X_train = df_train[feature_list].values
    X_test = df_test[feature_list].values
    assert(all_pos_finite(X_train))
    assert(all_pos_finite(X_test))

    # Make transformed features
    X_train = np.log(X_train)
    X_test = np.log(X_test)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    assert(all_finite(X_train))
    assert(all_finite(X_test))

    assert(target.endswith(metric))
    y_train = df_train[target].values
    y_test = df_test[target].values
    assert(all_finite(y_train))
    assert(all_finite(y_test))

    pred_tbl = btr.get_gauss_pred(X_train, y_train, X_test, methods)
    loss_tbl = btr.loss_table(pred_tbl, y_test, loss_dict)
    return loss_tbl


def run_experiment(df, metric, split_dict, all_features, target,
                   ref_method='GPR'):
    # TODO also do MLP
    k = 1.0 * RBF(length_scale=np.ones(len(all_features))) + \
        WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-3, np.inf))
    methods = {'iid': btr.JustNoise(), 'linear': BayesianRidge(),
               'GPR': GaussianProcessRegressor(kernel=k)}
    # Really dumb but GPR kernel needs to know D in advance
    k_sub = 1.0 * RBF(length_scale=np.ones(len(all_features) - 1)) + \
        WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-3, np.inf))

    summary = {}
    for split_name, splits in split_dict.iteritems():
        print split_name
        df_train, df_test, _ = ds.split_df(df, splits=splits)

        loss_tbl = \
            try_models(df_train, df_test, metric, all_features, target, methods)

        # Could also try just removing one
        L = [loss_tbl]
        for feature in all_features:
            mname = 'GPR-' + feature
            methods_sub = {mname: GaussianProcessRegressor(kernel=k_sub)}
            sub_f = list(all_features)
            sub_f.remove(feature)
            loss_tbl = try_models(df_train, df_test,
                                  metric, sub_f, target, methods_sub)
            L.append(loss_tbl)

        # Aggregate
        loss_tbl = pd.concat(L, axis=1)
        full_tbl = btr.loss_summary_table(loss_tbl, ref_method,
                                          pairwise_CI=True)
        # TODO use const names
        full_tbl[('NLL', 'mean')] -= full_tbl.loc[ref_method, ('NLL', 'mean')]
        summary[split_name] = full_tbl
    return summary

# TODO config
fname = '../../sampler-local/full_size/phase4/perf_sync.csv'

np.random.seed(56456)
do_plots = False

df = pd.read_csv(fname, header=0, index_col=None)

n_chains = df['n_chains'].max()
assert(n_chains == df['n_chains'].min())

agg_df = aggregate_df(df)
df = augment_df(df)

if do_plots:
    lb, ub = ESS_EB_fac(n_chains)
    plot_by(df, 'ESS', 'real_ess_mean', 'sampler', (lb, 1.0, ub))

    plot_by(df, 'eff', 'real_eff_mean', 'sampler')

    ax = df.boxplot('real_ness_mean', by='sampler', rot=90)
    ax.set_yscale('log')
    plt.xlabel('sampler')
    plt.ylabel('real_ness_mean')
    plt.show()

    ax = df.boxplot('real_eff_mean', by=['sampler', 'D'], rot=90)
    ax.set_yscale('log')
    plt.xlabel('sampler')
    plt.ylabel('real_ness_mean')
    plt.show()

split_dict = {'random': {ds.INDEX: (ds.RANDOM, 0.8)},
              'example': {'example': (ds.RANDOM, 0.8)}}

metric = 'mean'
target = 'real_essd_mean'
all_features = ['TG', 'TGR', 'ESS', 'D']

dumb_sampler = (df['sampler'] == 'emcee')
print dumb_sampler.sum()
small_ESS = (df['ESS'] <= 25)
print small_ESS.sum()
df_anal = df.loc[~(dumb_sampler | small_ESS), :]

missing = df_anal[all_features + [target]].isnull().any(axis=1)
df_anal = df_anal[~missing]
assert(not df_anal.isnull().any().any())

summary = run_experiment(df_anal, metric, split_dict, all_features, target)
for split, tbl in summary.iteritems():
    print split

    perf_tbl = just_format_it(tbl, shift_mod=3, unit_dict={'NLL': 'nats'},
                              non_finite_fmt={NAN_STR: '{--}'}, use_tex=True)
    print perf_tbl
