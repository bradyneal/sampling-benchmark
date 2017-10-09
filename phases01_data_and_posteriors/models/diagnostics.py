import pymc3 as pm
import numpy as np
from pymc3.backends.tracetab import trace_to_dataframe
from .utils import strace_gen
from functools import partial
import traceback


def get_diagnostics(trace, model):
    num_samples = len(trace)
    d1 = pm.diagnostics.gelman_rubin(trace)
    d2 = pm.diagnostics.effective_n(trace)
    d1['diagnostic'] = 'Gelman-Rubin'
    d2['diagnostic'] = 'ESS'
    try:
        max_corr = compute_max_corr_all(trace)
    except Exception:
        max_corr = traceback.format_exc()
    try:
        max_scale = compute_max_fisher_scale_all(trace, model)
    except Exception:
        max_scale = traceback.format_exc()
    return {
        'Gelman-Rubin': d1,
        'ESS': d2,
        'num_samples_per_chain': num_samples,
        'max_corr': max_corr,
        'max_scale': max_scale
    }


def compute_max_corr_all(trace):
    max_corrs = map(compute_max_corr, strace_gen(trace))
    max_corr = np.max(np.stack(max_corrs), axis=0)
    return max_corr


def compute_max_fisher_scale_all(trace, model):
    max_scales = map(partial(compute_max_fisher_scale, model=model),
                     strace_gen(trace))
    max_scale = np.nanmax(np.stack(max_scales), axis=0)
    return max_scale
    

def compute_max_corr(strace):
    df = trace_to_dataframe(strace)
    max_corr = max_corr = np.max(np.tril(np.abs(np.corrcoef(df.values, rowvar=0)), k=-1), axis=1)
    # redundant = max_corr > 0.99
    # red_names = df.columns.values[redundant]
    # print(red_names)
    return max_corr


def compute_max_fisher_scale(strace, model):
    epsilon = 1e-10
    n_checks = 20
    
    df = trace_to_dataframe(strace)
    
    idx_list = np.linspace(0, len(strace) - 1, n_checks, dtype=int)
    f = model.fastd2logp()
    
    # max_scale can be what we log
    max_scale = np.zeros(df.shape[1])
    for idx in idx_list:
        FI = f(strace[idx])
        FS = 0.5 * (FI + FI.T)  # sym to be safe
        Q, R = np.linalg.qr(FS)
        max_scale = np.fmax(max_scale, np.abs(np.diag(R)))
    # redundant = max_scale < epsilon
    # red_names = df.columns.values[redundant]
    # print(red_names)
    return max_scale
