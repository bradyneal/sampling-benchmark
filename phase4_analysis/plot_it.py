# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import METRICS_REF

from matplotlib import rcParams
c_cycle = rcParams['axes.color_cycle']


'''
gdf = df.groupby(['sampler', 'example'])
R = gdf.agg({'ks': np.mean, 'ESS': np.median, 'N': mp.max})
# TODO assert same as min
R['real_ess'] = ref / R['ks']
R['eff'] = R['real_ess'] / R['N']
'''

# TODO created augmented df for each metric that has all the quantities might want to plot
# then new df with aggregated by dim results

# Then clean plots:
#  ESS calib plots: ESS, ESS-N, EFF
# work out 95% regions under gauss
# add regression line

# plots wrt dimension
#  box plot for each ESS metric for dim x sampler
#   might need to subsample samplers if too busy

# do regr analysis per dim and agg
#   per dim first

# TODO do cross scatter plot of all the metrics
#   and show corr coef


def plot_ess_normed(df, metric, pooled=False):
    metric_ref = METRICS_REF[metric]
    metric = metric + '_pooled' if pooled else metric

    n_ref = df.groupby('example')['N'].median()

    plt.figure()
    gdf = df.groupby('sampler')
    for name, sdf in gdf:
        ref_val = sdf['example'].map(n_ref).values
        n_chains = sdf['n_chains'].values

        real_ess = metric_ref / sdf[metric].values
        estimated_ess = sdf['ESS'].values if pooled else \
            sdf['ESS'].values / n_chains
        plt.loglog(estimated_ess / ref_val, real_ess / ref_val, '.',
                   label=name, alpha=0.5)
    xgrid = np.logspace(-3.0, 3.0, 100)  # TODO automatic
    plt.loglog(xgrid, xgrid, 'k--')
    plt.legend(loc=3, ncol=4, fontsize=6,
               bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    plt.grid()
    plt.xlabel('normalized ESS')
    plt.ylabel('real normalized ESS')
    plt.title(metric)


def plot_ess(df, groupby_col, metric, pooled=False):
    metric_ref = METRICS_REF[metric]
    metric = metric + '_pooled' if pooled else metric

    plt.figure()
    gdf = df.groupby(groupby_col)
    for name, sub_df in gdf:
        n_chains = sub_df['n_chains'].values

        real_ess = metric_ref / sub_df[metric].values
        estimated_ess = sub_df['ESS'].values if pooled else \
            sub_df['ESS'].values / n_chains
        plt.loglog(estimated_ess, real_ess, '.', label=name, alpha=0.5)
    xgrid = np.logspace(0.0, 6.0, 100)  # TODO automatic
    plt.loglog(xgrid, xgrid, 'k--')
    plt.legend()
    plt.grid('on')
    plt.xlabel('ESS')
    plt.ylabel('real ESS')
    plt.title(metric)


def plot_eff(df, groupby_col, metric, pooled=False):
    metric_ref = METRICS_REF[metric]
    metric = metric + '_pooled' if pooled else metric

    plt.figure()
    gdf = df.groupby(groupby_col)
    for name, sub_df in gdf:
        n_chains = sub_df['n_chains'].values
        n_samples = sub_df['N'].values

        real_ess = metric_ref / sub_df[metric].values
        real_eff = real_ess / n_samples

        estimated_ess = sub_df['ESS'].values if pooled else \
            sub_df['ESS'].values / n_chains
        estimated_eff = estimated_ess / n_samples

        plt.loglog(estimated_eff, real_eff, '.', label=name, alpha=0.5)
    xgrid = np.logspace(-6.0, 0.0, 100)  # TODO automatic
    plt.loglog(xgrid, xgrid, 'k--')
    plt.legend(loc=3, ncol=4, fontsize=6,
               bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    plt.grid()
    plt.xlabel('estimated efficiency')
    plt.ylabel('real efficiency')
    plt.title(metric)


def plot_eff_vD(df, groupby_col, metric, pooled=False):
    metric_ref = METRICS_REF[metric]
    metric = metric + '_pooled' if pooled else metric

    plt.figure()
    gdf = df.groupby(groupby_col)
    for name, sub_df in gdf:
        n_samples = sub_df['N'].values

        real_ess = metric_ref / sub_df[metric].values
        real_eff = real_ess / n_samples

        D = sub_df['D'].values
        D = D + (np.random.rand(len(D)) - 0.5)

        plt.semilogy(D, real_eff, '.', label=name, alpha=0.5)
    plt.legend(loc=3, ncol=4, fontsize=6,
               bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    plt.grid()
    plt.xlabel('dimension')
    plt.ylabel('real efficiency')
    plt.title(metric)


# TODO config
#fname = '../../sampler-local/full_size/phase4/perf_sync.csv'

fname = '../perf_sync.csv'
df = pd.read_csv(fname, header=0, index_col=None)

plot_ess_normed(df, 'mean')


'''
examples = df['example'].unique()
samplers = df['sampler'].unique()

plt.figure()
for ex in examples:
    idx = df['example'] == ex
    real_ess = 1.0 / df.loc[idx, 'mean'].values
    plt.loglog(df.loc[idx, 'ESS'].values, real_ess, '.', label=ex)


plt.figure()
for sam in samplers:
    idx = df['sampler'] == sam
    df_sub = df.loc[idx, :]

    real_ess = 1.0 / df_sub['mean'].values
    n_samples = df_sub['N'].values
    n_chains = df_sub['n_chains'].values
    estimated_ess = df_sub['ESS'].values / n_chains
    plt.loglog(estimated_ess, real_ess, '.', label=sam, alpha=0.5)
xgrid = np.logspace(0.0, 6.0, 100)
plt.loglog(xgrid, xgrid, 'k--')
plt.legend()
plt.xlabel('ESS')
plt.ylabel('real ESS')
plt.title('mean')


plt.figure()
for sam in samplers:
    idx = df['sampler'] == sam
    df_sub = df.loc[idx, :]

    real_ess = 1.0 / df_sub['mean'].values
    n_samples = df_sub['N'].values
    n_chains = df_sub['n_chains'].values
    eff = real_ess / n_samples
    estimated_eff = df_sub['ESS'].values / (n_samples * n_chains)
    plt.loglog(estimated_eff, eff, '.', label=sam, alpha=0.5)
xgrid = np.logspace(-6.0, 0.0, 100)
plt.loglog(xgrid, xgrid, 'k--')
plt.legend()
plt.xlabel('estimated efficiency')
plt.ylabel('real efficiency')
plt.title('mean')

plt.figure()
for sam in samplers:
    idx = df['sampler'] == sam
    df_sub = df.loc[idx, :]

    real_ess = 1.0 / df_sub['mean_pooled'].values
    n_samples = df_sub['N'].values
    n_chains = df_sub['n_chains'].values
    eff = real_ess / (n_samples * n_chains)
    estimated_eff = df_sub['ESS'].values / (n_samples * n_chains)
    plt.loglog(estimated_eff, eff, '.', label=sam, alpha=0.5)
xgrid = np.logspace(-6.0, 0.0, 100)
plt.loglog(xgrid, xgrid, 'k--')
plt.legend()
plt.xlabel('estimated efficiency')
plt.ylabel('real efficiency')
plt.title('mean pooled')
'''
