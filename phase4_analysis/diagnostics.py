# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import pymc3 as pm


def combine_chains(chains):
    assert(all(np.ndim(x) == 2 for x in chains))
    min_n = min(x.shape[0] for x in chains)
    assert(min_n > 0)

    # Take end since these are the best samples, could also thin to size
    combo = np.stack([x[-min_n:, :] for x in chains], axis=2)
    # TODO permute to n_chains, n_steps, n_vars
    return combo


def geweke(chains):
    chains = combine_chains(chains)
    N, D, n_chains = chains.shape
    scores = []
    for nn in xrange(n_chains):
        for ii in xrange(D):
            R = pm.diagnostics.geweke(chains[:, ii, nn])
            scores.append(np.mean(R[:, 1]))
    # TODO look into what is best way to aggregate this into one score
    score = np.mean(scores)
    return score


def gelman_rubin(chains):
    chains = combine_chains(chains)
    Rhat = np.zeros(chains.shape[1])
    for ii in xrange(chains.shape[1]):
        x = chains[:, ii, :].T
        num_samples = x.shape[1]

        # Calculate between-chain variance
        B = num_samples * np.var(np.mean(x, axis=1), axis=0, ddof=1)

        # Calculate within-chain variance
        W = np.mean(np.var(x, axis=1, ddof=1), axis=0)

        # Estimate of marginal posterior variance
        Vhat = W * (num_samples - 1) / num_samples + B / num_samples

        Rhat[ii] = np.sqrt(Vhat / W)
    return np.mean(Rhat)  # TODO look into other combos


def effective_n(chains):
    chains = combine_chains(chains)

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

    def get_neff(x, Vhat):
        num_chains, num_samples = x.shape

        negative_autocorr = False
        t = 1

        rho = np.ones(num_samples)
        # Iterate until the sum of consecutive estimates of autocorrelation is
        # negative
        while not negative_autocorr and (t < num_samples):
            variogram = np.mean((x[:, t:] - x[:, :-t]) ** 2)
            rho[t] = 1.0 - variogram / (2.0 * Vhat)
            negative_autocorr = np.sum(rho[t - 1:t + 1]) < 0
            t += 1

        # TODO comment
        if t % 2:
            t -= 1

        n_eff = int(num_chains * num_samples / (1.0 + 2 * rho[1:t - 1].sum()))
        return min(num_chains * num_samples, n_eff)

    n_eff = np.zeros(chains.shape[1])
    for ii in xrange(chains.shape[1]):
        x = chains[:, ii, :].T

        Vhat = get_vhat(x)
        n_eff[ii] = get_neff(x, Vhat)
    return np.mean(n_eff)  # TODO look into other combos

# TODO add more
STD_DIAGNOSTICS = {'Geweke': geweke, 'Gelman_Rubin': gelman_rubin,
                   'ESS': effective_n}
