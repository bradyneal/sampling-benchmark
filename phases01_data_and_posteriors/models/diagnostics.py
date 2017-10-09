import pymc3 as pm
import numpy as np
from pymc3.backends.tracetab import trace_to_dataframe
from .utils import strace_gen


def get_diagnostics(trace):
    num_samples = len(trace)
    d1 = pm.diagnostics.gelman_rubin(trace)
    d2 = pm.diagnostics.effective_n(trace)
    d1['diagnostic'] = 'Gelman-Rubin'
    d2['diagnostic'] = 'ESS'
    max_corr = compute_max_stat_over_chains(compute_max_cor, trace)
    max_scale = compute_max_stat_over_chains(compute_max_fisher_scale, trace)
    return {
        'Gelman-Rubin': d1,
        'ESS': d2,
        'num_samples_per_chain': num_samples,
        'max_corr': max_corr,
        'max_scale': max_scale
    }


def compute_max_stat_over_chains(stat_fun, trace):
    return max(map(stat_fun, strace_gen(trace)))      
    

def compute_max_corr(strace):
    max_corr = np.max(np.tril(np.abs(np.corrcoef(df.values, rowvar=0)), k=-1), axis=1)
    # redundant = max_corr > 0.99
    # red_names = df.columns.values[redundant]
    # print(red_names)
    return max_corr


def compute_max_fisher_scale(strace):
    epsilon = 1e-10
    n_checks = 20
    
    df = trace_to_dataframe(tr)
    
    idx_list = np.linspace(0, len(tr) - 1, n_checks, dtype=int)
    f = model.fastd2logp()
    
    # max_scale can be what we log
    max_scale = np.zeros(df.shape[1])
    for idx in idx_list:
        FI = f(tr[idx])
        FS = 0.5 * (FI + FI.T)  # sym to be safe
        Q, R = np.linalg.qr(FS)
        max_scale = np.maximum(max_scale, np.abs(np.diag(R)))
    # redundant = max_scale < epsilon
    # red_names = df.columns.values[redundant]
    # print(red_names)
    return max_scale
