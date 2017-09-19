# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import pymc3 as pm


def geweke(chain):
    # TODO look into what is best way to aggregate this into one score
    N, D = chain.shape
    scores = []
    for ii in xrange(D):
        R = pm.diagnostics.geweke(chain[:, ii])
        scores.append(np.mean(R[:, 1]))
    score = np.mean(scores)
    return score

# TODO add more
STD_DIAGNOSTICS = {'Geweke': geweke}
