# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import scipy.stats as ss

DEFAULT_CLIP = 1.0


def mean(chain):
    assert(np.ndim(chain) == 2)
    return np.mean(chain, axis=0)


def var(chain):
    assert(np.ndim(chain) == 2)
    # var ddof=1 => unbiased
    return np.var(chain, axis=0, ddof=1)


def ks(exact, chain):
    assert(np.ndim(exact) == 2)
    D = exact.shape[1]
    assert(np.ndim(chain) == 2 and chain.shape[1] == D)

    ks_stat = np.array([ss.ks_2samp(exact[:, ii], chain[:, ii])[0]
                        for ii in xrange(D)])
    return ks_stat


MOMENT_METRICS = {'mean': mean, 'var': var}
OTHER_METRICS = {'ks': ks}

# Defined as expected loss for N(0,1) * n_samples
METRICS_REF = {'mean': 1.0, 'var': 2.0, 'ks': 0.822}


def rectified_sq_error(exact, approx, clip=DEFAULT_CLIP):
    # Debatable if rectifier (min) should be before or after sum
    err = np.mean(np.minimum(clip, (exact - approx) ** 2))
    return err


def eval_inc(exact_chain, all_chains, metric, all_idx):
    n_grid, n_chains = all_idx.shape
    assert(n_chains >= 1)
    assert(len(all_chains) == n_chains)
    D = all_chains[0].shape[1]

    if metric in MOMENT_METRICS:
        estimator = MOMENT_METRICS[metric]
        exact = estimator(exact_chain)
        moment_metric = True
    else:
        assert(metric in OTHER_METRICS)
        estimator = OTHER_METRICS[metric]
        moment_metric = False

    err = np.zeros((n_grid, n_chains))
    for c_num, chain in enumerate(all_chains):
        assert(chain.ndim == 2 and chain.shape[1] == D)
        # Just do it naively right now instead of online & incremental
        for ii, n_samples in enumerate(all_idx[:, c_num]):
            if moment_metric:
                approx = estimator(chain[:n_samples, :])
                err[ii, c_num] = rectified_sq_error(exact, approx)
            else:
                approx = estimator(exact_chain, chain[:n_samples, :])
                err[ii, c_num] = rectified_sq_error(0.0, approx)
    err = np.mean(err, axis=1)  # ave over chains
    assert(err.shape == (n_grid,))
    return err
