# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np


def _sq_err(exact, approx):
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
