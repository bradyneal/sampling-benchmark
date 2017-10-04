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


def plot_by(df, x, y, by):
    plt.figure()
    gdf = df.groupby(by)
    for name, sdf in gdf:
        plt.loglog(sdf[x].values, sdf[y].values, '.', label=name, alpha=0.5)
    xgrid = np.logspace(np.log10(df[x].min()), np.log10(df[x].max()), 100)
    plt.loglog(xgrid, xgrid, 'k--')
    plt.legend(loc=3, ncol=4, fontsize=6,
               bbox_to_anchor=(0.0, 1.02, 1.0, 0.102))
    plt.grid()
    plt.xlabel(x)
    plt.ylabel(y)


# TODO config
#fname = '../../sampler-local/full_size/phase4/perf_sync.csv'

fname = '../perf_sync.csv'
df = pd.read_csv(fname, header=0, index_col=None)

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
    df['real_ess_' + metric] = metric_ref / df[metric]
    df['real_ness_' + metric] = df['real_ess_' + metric] / df['n_ref']
    df['real_eff_' + metric] = df['real_ess_' + metric] / df['N']

    metric = metric + '_pooled'
    df['real_ess_' + metric] = metric_ref / df[metric]
    df['real_ness_' + metric] = df['real_ess_' + metric] / df['n_ref']
    total_samples = df['N'] * df['n_chains']
    df['real_eff_' + metric] = df['real_ess_' + metric] / total_samples

plot_by(df, 'eff', 'real_eff_mean', 'sampler')
