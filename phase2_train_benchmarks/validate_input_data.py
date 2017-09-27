# Ryan Turner (turnerry@iro.umontreal.ca)
import os
import numpy as np
import scipy.stats as ss
import ConfigParser
import fileio as io


def moments_report(X):
    V = np.var(X, axis=0)
    var_ratio = np.log10(np.max(V) / np.min(V))

    C = np.cov(X, rowvar=0)
    cond_number = np.log10(np.linalg.cond(C))

    max_skew = np.max(np.abs(ss.skew(X, axis=0)))
    max_kurt = np.max(ss.kurtosis(X, axis=0))

    print 'log10 var ratio %f, cond numer %f' % (var_ratio, cond_number)
    print 'max skew %f, max kurt %f' % (max_skew, max_kurt)

EPSILON = 1e-12

config_file = '../config.ini'
config = ConfigParser.RawConfigParser()
#assert(os.path.isabs(config_file))
config.read(config_file)
input_path = io.abspath2(config.get('phase1', 'output_path'))
data_ext = config.get('common', 'csv_ext')
sep = '_'

print 'searching for input data in'
print input_path
chain_files = sorted(f for f in os.listdir(input_path) if f.endswith(data_ext))
print 'found %d files' % len(chain_files)

for chain in chain_files:
    print '-' * 20
    X = io.load_np(input_path, chain, '')
    assert(X.ndim == 2)
    N, D = X.shape

    finite = np.all(np.isfinite(X))

    acc = np.abs(np.diff(X, axis=0)) > EPSILON
    acc_valid = np.all(np.any(acc, 1) == np.all(acc, 1))
    acc_rate = np.mean(acc[:, 0])

    print chain
    print 'N = %d, D = %d' % (N, D)
    print 'finite %d, accept %d' % (finite, acc_valid)
    print 'acc rate %f' % acc_rate

    print 'full data'
    moments_report(X)

    print '2nd half data'
    moments_report(X[int(N / 2):, :])
print 'done'
