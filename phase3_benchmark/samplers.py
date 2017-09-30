# Ryan Turner (turnerry@iro.umontreal.ca)
import pymc3 as pm
from emcee import EnsembleSampler

# See:
# http://docs.pymc.io/api/inference.html#step-methods

def test_case_mix():
    steps = [pm.NUTS(), pm.Metropolis()]
    return steps


def cauchy():
    return pm.Metropolis(proposal_dist=pm.CauchyProposal)


def laplace():
    return pm.Metropolis(proposal_dist=pm.LaplaceProposal)


BUILD_STEP_PM = {'NUTS-default': pm.NUTS,
                 'Metro-default': pm.Metropolis,
                 'slice-default': pm.Slice,
                 'HMC-default': pm.HamiltonianMC,
                 'Cauchy-proposal': cauchy,
                 'Laplace-proposal': laplace,
                 'mix': test_case_mix}

BUILD_STEP_MC = {'emcee': EnsembleSampler}

assert(set(BUILD_STEP_PM.keys()).isdisjoint(BUILD_STEP_MC.keys()))
