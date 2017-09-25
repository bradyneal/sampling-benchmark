# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np


def mean(exact_chain, sampler_chain):
    exact_mu = np.mean(exact_chain, axis=0)
    approx_mu = np.mean(sampler_chain, axis=0)

    V = np.var(exact_chain, axis=0)  # TODO look into bias arg on all var calls
    N = sampler_chain.shape[0]

    err = (exact_mu - approx_mu) ** 2
    ess = V / err  # Cap at N??
    eff = ess / N
    # Do we want the sqrt??
    return np.sqrt(np.mean(err)), np.mean(ess), np.mean(eff)


def var(exact_chain, sampler_chain):
    exact_var = np.var(exact_chain, axis=0)
    approx_var = np.var(sampler_chain, axis=0)

    V = np.var(exact_chain, axis=0)
    N = sampler_chain.shape[0]

    err = (exact_var - approx_var) ** 2
    ess = 2.0 * (V ** 2) / err
    eff = ess / N
    return np.sqrt(np.mean(err)), np.mean(ess), np.mean(eff)

# TODO add ks and cov
STD_METRICS = {'mean': mean, 'var': var}


def eval_inc(exact_chain, curr_chain, metric, idx):
    assert(np.ndim(idx) == 1)
    metric_f = STD_METRICS[metric]

    # Just so it naively right now
    err = np.zeros(len(idx))
    ess = np.zeros(len(idx))
    eff = np.zeros(len(idx))
    for ii, n_samples in enumerate(idx):
        err[ii], ess[ii], eff[ii] = \
           metric_f(exact_chain, curr_chain[:n_samples, :])
    return err, ess, eff
