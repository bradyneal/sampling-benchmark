# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import BayesianRidge
from metrics import METRICS_REF

import bt.benchmark_tools_regr as btr
import bt.data_splitter as ds

import matplotlib.pyplot as plt
from matplotlib import rcParams
c_cycle = rcParams['axes.color_cycle']

DIAGS = ('Geweke', 'ESS', 'Gelman_Rubin')  # TODO import from elsewhere

# TODO re-run with no clip in perf
#   also adjust MIN_ESS to per chain

# do regr analysis per dim and agg
#   per dim first
# TODO look at scatter and see which form is most gaussian
#  log-scale, ess, eff, err, log-eff

# TODO do cross scatter plot of all the metrics
#   and show corr coef


def ESS_EB_fac(n_estimates, conf=0.95):
    P_ub = 0.5 * (1.0 + conf)
    P_lb = 1.0 - P_ub

    lb_fac = n_estimates / ss.chi2.ppf(P_ub, n_estimates)
    ub_fac = n_estimates / ss.chi2.ppf(P_lb, n_estimates)
    return lb_fac, ub_fac


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

    for metric in sorted(METRICS_REF.keys()):
        metric_ref = METRICS_REF[metric]
        # TODO add jac factors to each of these for pred lik scores
        df['real_ess_' + metric] = metric_ref / df[metric]
        df['real_ness_' + metric] = df['real_ess_' + metric] / df['n_ref']
        df['real_eff_' + metric] = df['real_ess_' + metric] / df['N']

        metric = metric + '_pooled'
        df['real_ess_' + metric] = metric_ref / df[metric]
        df['real_ness_' + metric] = df['real_ess_' + metric] / df['n_ref']
        total_samples = df['N'] * df['n_chains']
        df['real_eff_' + metric] = df['real_ess_' + metric] / total_samples
    return df  # return anyway


def plot_by(df, x, y, by, fac_lines=(1.0,)):
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


def try_models(df_train, df_test, metric, feature_list, target_list, methods):
    loss_dict = {'NLL': btr.log_loss}

    X_train = df_train[feature_list].values
    assert(all_pos_finite(X_train))
    X_test = df_test[feature_list].values
    assert(all_pos_finite(X_test))
    # TODO augment with log-scale, inv-scale??
    # TODO remove lines with nan features

    summary = {}
    for target in target_list:
        assert(target.endswith(metric))
        y_train = df_train[target].values
        assert(all_pos_finite(y_train))
        y_test = df_test[target].values
        assert(all_pos_finite(y_test))

        # TODO robust standardize data, need to adjust jac
        # TODO use const for metric

        pred_tbl = btr.get_gauss_pred(X_train, y_train, X_test, methods)
        loss_tbl = btr.loss_table(pred_tbl, y_test, loss_dict)
        loss_tbl = loss_tbl.xs('NLL', axis=1, level='metric', drop_level=True)
        summary[target] = loss_tbl.mean(axis=0)

        pred_tbl = btr.get_gauss_pred(X_train, np.log(y_train), X_test, methods)
        loss_tbl = btr.loss_table(pred_tbl, np.log(y_test), loss_dict)
        loss_tbl = loss_tbl.xs('NLL', axis=1, level='metric', drop_level=True)
        summary['log_' + target] = loss_tbl.mean(axis=0)

        # TODO get mean delta to ref_method
        # TODO use jacobian to xform
    summary = pd.DataFrame(summary)
    return summary


def run_experiment(df, metric, methods, ref_method, split_dict):
    common = ['D', 'N']
    diag_list = ['TG', 'TGR', 'ESS']
    target_list = [c for c in df.columns if c.endswith(metric)]

    all_features = diag_list + common
    # TODO figure out why some missing
    missing = df[all_features].isnull().any(axis=1)
    df = df[~missing]
    assert(not df[all_features].isnull().any().any())

    summary = {}
    for split_name, splits in split_dict.iteritems():
        df_train, df_test, _ = ds.split_df(df, splits=splits)

        for diag in diag_list:
            feature_list = [diag] + common
            summary[(split_name, diag)] = try_models(df_train, df_test, metric, feature_list, target_list, methods)

        # Now try all:
        summary[(split_name, 'all')] = try_models(df_train, df_test, metric, all_features, target_list, methods)
    summary = pd.concat(summary, axis=1)
    return summary

# TODO try after tossing out emcee

# TODO config
fname = '../../sampler-local/full_size/phase4/perf_sync.csv'

np.random.seed(56456)

#fname = '../perf_sync.csv'
df = pd.read_csv(fname, header=0, index_col=None)

n_chains = df['n_chains'].max()
assert(n_chains == df['n_chains'].min())

agg_df = aggregate_df(df)
df = augment_df(df)

df['TG'] = df['Geweke'].abs()
df['TGR'] = (df['Gelman_Rubin'] - 1.0).abs()


'''
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
'''

split_dict = {'sampler': {'sampler': (ds.RANDOM, 0.8)},
              'example': {'example': (ds.RANDOM, 0.8)},
              'joint': {'sampler': (ds.RANDOM, 0.8), 'example': (ds.RANDOM, 0.8)}}

# TODO add GPR
methods = {'iid': btr.JustNoise(), 'linear': BayesianRidge()}
ref_method = 'iid'

summary = run_experiment(df, 'mean', methods, ref_method, split_dict)

#   start with just mean metric, but check results don't change much with others
#   try sklearn linear (L1+L2) models, and GPs/ANNs next
#   could do MCMC hyper-params to be consistent
#   add gaussian predictors to bt
#   if linear models competetive, find top perf space and feed into stats models
#   to get stat results
