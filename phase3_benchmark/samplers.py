# Ryan Turner (turnerry@iro.umontreal.ca)
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

BUILD_STEP_PM = {'NUTS-default': NUTS,
                 'Metro-default': metro_default,
                 'Cauchy-proposal': cauchy,
                 'Laplace-proposal': laplace,
                 'mix': test_case_mix}

BUILD_STEP_MC = {'emcee': EnsembleSampler}

assert(set(BUILD_STEP_PM.keys()).isdisjoint(BUILD_STEP_MC.keys()))
