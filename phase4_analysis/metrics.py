# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np


def _sq_err(exact, approx):
    # TODO rename rmse and implement MAPE
    err = np.sqrt(np.sum((exact - approx) ** 2))
    return err


def mean(exact_chain, sampler_chain):
    exact_mu = np.mean(exact_chain, axis=0)
    sampler_mu = np.mean(sampler_chain, axis=0)
    err = _sq_err(exact_mu, sampler_mu)
    return err


def std(exact_chain, sampler_chain):
    exact_std = np.std(exact_chain, axis=0)
    sampler_std = np.std(sampler_chain, axis=0)
    err = _sq_err(exact_std, sampler_std)
    return err

STD_METRICS = {'mean': mean, 'std': std}


def build_target(exact_chain, metric):
    # This will be phased out when we go to n_eff
    metric_f = STD_METRICS[metric]

    N = exact_chain.shape[0] // 2
    target = metric_f(exact_chain, exact_chain[:N, :])
    return target


def eval_inc(exact_chain, curr_chain, metric, idx):
    assert(np.ndim(idx) == 1)
    metric_f = STD_METRICS[metric]

    # Just so it naively right now
    perf = np.zeros(len(idx))
    for ii, n_samples in enumerate(idx):
        perf[ii] = metric_f(exact_chain, curr_chain[:n_samples, :])
    return perf
