# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np

DEFAULT_CLIP = 1.0


def mean(chain):
    assert(np.ndim(chain) == 2)
    return np.mean(chain, axis=0)


def var(chain):
    assert(np.ndim(chain) == 2)
    # var ddof=1 => unbiased
    return np.var(chain, axis=0, ddof=1)

STD_METRICS = {'mean': mean, 'var': var}


def rectified_sq_error(exact, approx, clip=DEFAULT_CLIP):
    # Debatable if rectifier (min) should be before or after sum
    err = np.sum(np.minimum(clip, (exact - approx) ** 2))
    return err


def eval_inc(exact_chain, all_chains, metric, all_idx):
    n_grid, n_chains = all_idx.shape
    assert(n_chains >= 1)
    assert(len(all_chains) == n_chains)
    D = all_chains[0].shape[1]

    estimator = STD_METRICS[metric]
    exact = estimator(exact_chain)

    err = np.zeros((n_grid, n_chains))
    for c_num, chain in enumerate(all_chains):
        assert(chain.ndim == 2 and chain.shape[1] == D)
        # Just do it naively right now instead of online & incremental
        for ii, n_samples in enumerate(all_idx[:, c_num]):
            approx = estimator(chain[:n_samples, :])
            err[ii, c_num] = rectified_sq_error(exact, approx)
    err = np.mean(err, axis=1)  # ave over chains
    assert(err.shape == (n_grid,))
    return err
