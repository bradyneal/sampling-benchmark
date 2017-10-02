# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import scipy.stats as ss
import pymc3 as pm

MIN_ESS = 1.0


def summaries(x):
    D = {'mean': np.mean(x), 'median': np.median(x),
         'min': np.min(x), 'max': np.max(x),
         'gmean': ss.gmean(x), 'hmean': ss.hmean(x)}
    return D


def geweke(chains, max_intervals=20):
    n_chains, N, D = chains.shape
    scores = np.zeros((n_chains, D))
    for nn in xrange(n_chains):
        for ii in xrange(D):
            # Will this formula never lead to too many intervals??
            intervals = min(max_intervals, N // 10) + 1
            R = pm.diagnostics.geweke(chains[nn, :, ii], intervals=intervals)
            # assert(R.shape[0] == intervals)
            # multiply by sqrt so expected to follow normal
            # Could also try corrected min p-value
            scores[nn, ii] = np.mean(R[:, 1]) * np.sqrt(intervals)
    scores = np.mean(scores, axis=0) * np.sqrt(n_chains)
    return scores


def gelman_rubin(chains):
    n_chains, num_samples, D = chains.shape
    Rhat = np.zeros(D)
    for ii in xrange(D):
        x = chains[:, :, ii]

        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        Rhat[ii] = np.sqrt(Vhat / W)
    return Rhat


def effective_n(chains):
    n_chains, num_samples, D = chains.shape

    def get_vhat(x):
        # TODO eliminate repetition with gelman rubin
        num_samples = x.shape[1]

        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        return Vhat

    def get_neff(x, Vhat, inc_frac=1.05):
        num_chains, num_samples = x.shape

        negative_autocorr = False
        t = 1

        rho = np.ones(num_samples)
        # Iterate until the sum of consecutive estimates of autocor is neg.
        # TODO emcee uses FFT to estimate this for speed. Look into it.
        while not negative_autocorr and (t < num_samples):
            variogram = np.mean((x[:, t:] - x[:, :-t]) ** 2)
            rho[t] = 1.0 - variogram / (2.0 * Vhat)
            negative_autocorr = np.sum(rho[t - 1:t + 1]) < 0
            # original was t += 1, but that just gets too slow
            t_new = int(np.ceil(t * inc_frac))
            assert(t_new > t)  # prevent infinite loop
            t = t_new

        # TODO comment
        if t % 2:
            t -= 1

        n_eff = num_chains * num_samples / (1.0 + 2 * rho[1:t - 1].sum())
        # If we want to cap it:
        # n_eff = min(num_chains * num_samples + 0.0, n_eff)
        return n_eff

    n_eff = np.zeros(D)
    for ii in xrange(D):
        x = chains[:, :, ii]

        Vhat = get_vhat(x)
        n_eff[ii] = get_neff(x, Vhat)
    n_eff = np.maximum(MIN_ESS, n_eff)
    return n_eff

ESS = 'ESS'
STD_DIAGNOSTICS = {'Geweke': geweke, 'Gelman_Rubin': gelman_rubin,
                   ESS: effective_n}
