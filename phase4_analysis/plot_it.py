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

from matplotlib import rcParams, use
use('pdf')
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.family'] = 'STIXGeneral'
import matplotlib.pyplot as plt

DIAGS = ('Geweke', 'ESS', 'Gelman_Rubin')  # TODO import from elsewhere

# TODO run with both perf agg that has had rectified and unrectified sq loss
# TODO save figs to output dir in config


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


def augment_df(df, success_thold=12):
    # Note: this is happening inplace!
    n_ref = df.groupby('example')['N'].median()
    df['n_ref'] = df['example'].map(n_ref)
    df['short_name'] = df['sampler'].str.split('-').str[0]

    df['ESS_pooled'] = df['ESS']
    df['ESS'] = df['ESS_pooled'] / df['n_chains']
    df['success'] = df['ESS'] > success_thold

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
        df['real_essd_' + metric] = \
            ESS_deviation(df[metric].values,
                          df['ESS'].values, df['n_chains'].values, metric_ref)
        df['success_' + metric] = df['real_ess_' + metric] > success_thold

        metric = metric + '_pooled'
        df['real_ess_' + metric] = metric_ref / df[metric]
        df['real_ness_' + metric] = df['real_ess_' + metric] / df['n_ref']
        total_samples = df['N'] * df['n_chains']
        df['real_eff_' + metric] = df['real_ess_' + metric] / total_samples
        df['real_essd_' + metric] = \
            ESS_deviation(df[metric].values, df['ESS_pooled'].values,
                          1.0, metric_ref)
    return df  # return anyway


def plot_by(df, x, y, by, fac_lines=(1.0,)):
    # TODO make logscale arg
    fig = plt.figure(figsize=(3.5, 3.5), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.7])

    gdf = df.groupby(by)
    for name, sdf in gdf:
        ax.loglog(sdf[x].values, sdf[y].values, '.', label=name, alpha=0.5)
    lower, upper = ax.get_xlim()
    xgrid = np.logspace(np.log10(lower), np.log10(upper), 100)
    for f in fac_lines:
        style = 'k--' if f == 1 else 'k:'
        ax.loglog(xgrid, f * xgrid, style)
    plt.legend(loc=3, ncol=5, fontsize=6,
               bbox_to_anchor=(0.0, 1.02, 0.9, 0.102))
    plt.grid()
    ax.tick_params(labelsize=6)
    #plt.xlabel(x, fontsize=10)
    #plt.ylabel(y, fontsize=10)
    return ax, xgrid


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
    y_train = np.clip(df_train[target].values, -5, 5)
    y_test = np.clip(df_test[target].values, -5, 5)
    assert(all_finite(y_train))
    assert(all_finite(y_test))

    pred_tbl = btr.get_gauss_pred(X_train, y_train, X_test, methods)
    loss_tbl = btr.loss_table(pred_tbl, y_test, loss_dict)
    return loss_tbl, scaler


def run_experiment(df, metric, split_dict, all_features, target,
                   ref_method='GPR'):
    # TODO also do MLP
    summary = {}
    ref_model = {}
    for split_name, splits in split_dict.iteritems():
        print 'running', split_name
        df_train, df_test, _ = ds.split_df(df, splits=splits)

        k = 1.0 * RBF(length_scale=np.ones(len(all_features))) + \
            WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-3, np.inf))
        methods = {'iid': btr.JustNoise(), 'linear': BayesianRidge(),
                   'GPR': GaussianProcessRegressor(kernel=k)}
        loss_tbl, scaler = try_models(df_train, df_test, metric,
                                      all_features, target, methods)
        ref_model[split_name] = (methods[ref_method], scaler)

        # Could also try just removing one
        L = [loss_tbl]
        for feature in all_features:
            mname = 'GPR-' + feature
            k_sub = 1.0 * RBF(length_scale=np.ones(len(all_features) - 1)) + \
                WhiteKernel(noise_level=0.1**2,
                            noise_level_bounds=(1e-3, np.inf))
            methods_sub = {mname: GaussianProcessRegressor(kernel=k_sub)}
            sub_f = list(all_features)
            sub_f.remove(feature)
            loss_tbl, _ = try_models(df_train, df_test,
                                     metric, sub_f, target, methods_sub)
            L.append(loss_tbl)

        # Aggregate
        loss_tbl = pd.concat(L, axis=1)
        full_tbl = btr.loss_summary_table(loss_tbl, ref_method,
                                          pairwise_CI=True)
        # TODO use const names
        full_tbl[('NLL', 'mean')] -= full_tbl.loc[ref_method, ('NLL', 'mean')]
        summary[split_name] = full_tbl
    return summary, ref_model


def ecdf(x):
    plt.plot(np.sort(x), np.linspace(0, 1, len(x), endpoint=False))


def NESS_tbl_tex(df):
    metric_list = sorted(METRICS_REF.keys())
    gdf = df.groupby('sampler')
    cols = ['success_' + m for m in metric_list]
    success_tbl = gdf[cols].mean()
    NESS_tbl = gdf[['real_ness_' + m for m in metric_list]].mean()
    rdf = pd.concat((NESS_tbl, success_tbl), axis=1)
    return rdf.to_latex(float_format='%.3f')


def make_all_plots(df):
    metric_clean = {'ks': 'KS', 'mean': r'$\mu$', 'var': r'$\sigma^2$'}

    n_chains = df['n_chains'].max()
    if n_chains != df['n_chains'].min():
        print 'warning! error bars assume n_chains constant'
    lb, ub = ESS_EB_fac(n_chains)
    metric_list = sorted(METRICS_REF.keys())

    # Calibration plots
    df_sub = df[(df['n_chains'] == n_chains) & (df['success'])]
    for metric in metric_list:
        ax, xgrid = plot_by(df_sub, 'ESS', 'real_ess_' + metric,
                            'short_name', (lb, 1.0, ub))
        ax.set_ylim(top=1.05 * np.max(ub * xgrid))
        ax.set_xlim(xgrid[0], xgrid[-1])
        plt.xlabel('ESS')
        plt.ylabel('RESS ' + metric_clean[metric], fontsize=10)
        plt.savefig('real_ess_' + metric + '.pdf', dpi=300, format='pdf')

        ax, xgrid = plot_by(df_sub, 'eff', 'real_eff_' + metric, 'short_name')
        plt.xlabel('ESS / $N$')
        plt.ylabel('EFF ' + metric_clean[metric], fontsize=10)
        ax.set_xscale('linear')
        ax.set_ylim(top=1.05 * np.max(ub * xgrid))
        ax.set_xlim(xgrid[0], xgrid[-1])
        plt.savefig('real_eff_' + metric + '.pdf', dpi=300, format='pdf')

    # Box plots
    for metric in metric_list:
        df_sub = df[df['success_' + metric]]

        fig = plt.figure(figsize=(3.5, 3.5), dpi=80,
                         facecolor='w', edgecolor='k')
        ax = fig.add_axes([0.15, 0.2, 0.8, 0.75])
        df_sub.boxplot('real_ness_' + metric, by='short_name', rot=45, ax=ax)
        plt.title('')
        plt.suptitle('')
        ax.set_yscale('log')
        ax.set_ylim(top=20)
        ax.tick_params(labelsize=6)
        plt.xlabel('sampler', fontsize=10)
        plt.ylabel('NESS ' + metric_clean[metric], fontsize=10)
        plt.show()
        plt.savefig('ness_' + metric + '.pdf', dpi=300, format='pdf')

        fig = plt.figure(figsize=(3.5, 3.5), dpi=80,
                         facecolor='w', edgecolor='k')
        ax = fig.add_axes([0.15, 0.2, 0.8, 0.75])
        df_sub.boxplot('real_eff_' + metric, by='short_name', rot=45, ax=ax)
        plt.title('')
        plt.suptitle('')
        ax.set_yscale('log')
        ax.set_ylim(top=20)
        ax.tick_params(labelsize=6)
        plt.xlabel('sampler', fontsize=10)
        plt.ylabel('EFF ' + metric_clean[metric], fontsize=10)
        plt.show()
        plt.savefig('box_eff_' + metric + '.pdf', dpi=300, format='pdf')


def meta_analysis_tex(df):
    split_dict = {'random': {ds.INDEX: (ds.RANDOM, 0.8)},
                  'example': {'example': (ds.RANDOM, 0.8)}}

    metric = 'mean'
    target = 'real_essd_mean'
    all_features = ['TG', 'TGR', 'ESS', 'D']

    df_anal = df[df['success']]

    missing = df_anal[all_features + [target]].isnull().any(axis=1)
    df_anal = df_anal[~missing]
    assert(not df_anal.isnull().any().any())

    summary, ref_model = run_experiment(df_anal, metric, split_dict,
                                        all_features, target)
    perf_tex = {}
    for split, tbl in summary.iteritems():
        perf_tbl = just_format_it(tbl, shift_mod=3, unit_dict={'NLL': 'nats'},
                                  non_finite_fmt={NAN_STR: '{--}'},
                                  use_tex=True)
        perf_tex[split] = perf_tbl
    return perf_tex

# TODO config
fname = '../../sampler-local/cedar1_P4/perf_sync.csv'

np.random.seed(56456)
do_plots = True

df = pd.read_csv(fname, header=0, index_col=None)
df = df[df['n_chains'] > 1]  # Filter out where no way to average error

agg_df = aggregate_df(df)
df = augment_df(df)

# Overall performance table
print NESS_tbl_tex(df)

# All of the plots
make_all_plots(df)

# meta-analysis
tbl_dict = meta_analysis_tex(df)
for k, tbl in tbl_dict.iteritems():
    print k
    print tbl
print 'done'
