# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import pymc3 as pm
from emcee import EnsembleSampler

# See:
# http://docs.pymc.io/api/inference.html#step-methods

def test_case_mix(step_kwds):
    steps = [pm.NUTS(), pm.Metropolis()]
    return steps


def cauchy(step_kwds):
    return pm.Metropolis(proposal_dist=pm.CauchyProposal, **step_kwds)


def laplace(step_kwds):
    return pm.Metropolis(proposal_dist=pm.LaplaceProposal, **step_kwds)


def metro_default(step_kwds):
    return pm.Metropolis(**step_kwds)


def NUTS(step_kwds):
    return pm.NUTS()  # Add options later??


def HMC(step_kwds):
    if 'scaling' in step_kwds:
        # Convert to cov
        step_kwds['scaling'] = step_kwds['scaling'] ** 2
        assert('is_cov' not in step_kwds)
        step_kwds['is_cov'] = True
    return pm.HamiltonianMC(**step_kwds)


def slice_default(step_kwds):
    if 'scaling' in step_kwds:
        step_kwds['w'] = np.maximum(1.0, step_kwds['scaling'])
        del step_kwds['scaling']
    step_kwds['iter_limit'] = 10 ** 6
    return pm.Slice(**step_kwds)

BUILD_STEP_PM = {'NUTS-default': NUTS,
                 'Metro-default': metro_default,
                 'Cauchy-proposal': cauchy,
                 'Laplace-proposal': laplace,
                 'mix': test_case_mix,
                 'HMC-default': HMC,
                 'slice-default': slice_default}

BUILD_STEP_MC = {'emcee': EnsembleSampler}

assert(set(BUILD_STEP_PM.keys()).isdisjoint(BUILD_STEP_MC.keys()))
