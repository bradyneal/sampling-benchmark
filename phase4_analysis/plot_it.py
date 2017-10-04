import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
gdf = df.groupby(['sampler', 'example'])
R = gdf.agg({'ks': np.mean, 'ESS': np.median, 'N': mp.max})
# TODO assert same as min
R['real_ess'] = ref / R['ks']
R['eff'] = R['real_ess'] / R['N']
'''


# TODO config
fname = '../../sampler-local/full_size/phase4/perf_sync.csv'
df = pd.read_csv(fname, header=0, index_col=None)

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
