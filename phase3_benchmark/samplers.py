# Ryan Turner (turnerry@iro.umontreal.ca)
import pymc3 as pm

# See:
# http://docs.pymc.io/api/inference.html#step-methods

def test_case_mix():
    steps = [pm.NUTS(), pm.Metropolis()]
    return steps

BUILD_STEP = {'NUTS-default': pm.NUTS,
              'Metro-default': pm.Metropolis,
              'mix': test_case_mix}
