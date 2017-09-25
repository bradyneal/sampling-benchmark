# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np

# var ddof=1 => unbiased


def mean(exact_chain, sampler_chain):
    exact_mu = np.mean(exact_chain, axis=0)
    approx_mu = np.mean(sampler_chain, axis=0)

    err = (exact_mu - approx_mu) ** 2
    return err


def mean_ref(exact_chain):
    ref = np.var(exact_chain, axis=0, ddof=1)
    return ref


def var(exact_chain, sampler_chain):
    exact_var = np.var(exact_chain, axis=0, ddof=1)
    approx_var = np.var(sampler_chain, axis=0, ddof=1)

    err = (exact_var - approx_var) ** 2
    return err


def var_ref(exact_chain):
    ref = 2.0 * np.var(exact_chain, axis=0, ddof=1) ** 2
    return ref


# TODO add ks and cov
STD_METRICS = {'mean': mean, 'var': var}
STD_METRICS_REF = {'mean': mean_ref, 'var': var_ref}
# TODO assert same keys


def eval_inc(exact_chain, all_chains, metric, all_idx):
    n_grid, n_chains = all_idx.shape
    assert(n_chains >= 1)
    assert(len(all_chains) == n_chains)
    D = all_chains[0].shape[1]

    metric_f = STD_METRICS[metric]

    ref_num = STD_METRICS_REF[metric](exact_chain)
    assert(ref_num.shape == (D,))
    n_ave = np.mean(all_idx, axis=1)
    assert(n_ave.shape == (n_grid,))

    err = np.zeros((n_grid, D, n_chains))
    for c_num, chain in enumerate(all_chains):
        assert(chain.ndim == 2 and chain.shape[1] == D)
        # Just so it naively right now instead of online & incremental
        for ii, n_samples in enumerate(all_idx[:, c_num]):
            err[ii, :, c_num] = metric_f(exact_chain, chain[:n_samples, :])
    err = np.mean(err, axis=2)  # ave over chains
    assert(err.shape == (n_grid, D))

    err_agg = np.mean(err, axis=1)
    ess = np.mean(ref_num[None, :] / err, axis=1)
    # TODO consider best denominator for N, make so exact has E[eff]=1
    eff = ess / n_ave
    return err_agg, ess, eff
