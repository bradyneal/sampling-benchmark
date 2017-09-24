# Ryan Turner (turnerry@iro.umontreal.ca)
import cPickle as pkl
import os
import numpy as np
import scipy.stats as ss
import theano
import theano.tensor as T
import phase2_train_benchmarks.model_wrappers as p2
import phase3_benchmark.models as p3

# This requires:
# export PYTHONPATH=./phase2_train_benchmarks/bench_models/nade/:$PYTHONPATH
# due to crap in __init__ in nade
# TODO fix

# TODO put in a main func

def test_model(input_path, params_file, N):
    model_file = os.path.join(input_path, params_file)
    print 'loading %s' % model_file
    with open(model_file, 'rb') as f:
        model_name, D, params_dict = pkl.load(f)

    # Sample some data
    X = p3.SAMPLE_MODEL[model_name](params_dict, N=N)
    assert(X.shape == (N, D))

    p2_model_class = p2.STD_BENCH_MODELS[model_name]
    v0 = p2_model_class.loglik_chk(X, params_dict)
    v1 = np.array([p2_model_class.loglik_chk(X[ii, None], params_dict)[0]
                   for ii in xrange(N)])
    print 'err1 %f' % np.log10(np.max(np.abs(v0 - v1)))

    x_tt = T.vector()
    logpdf_tt = p3.BUILD_MODEL[model_name](x_tt, params_dict)
    logpdf_f = theano.function([x_tt], logpdf_tt)

    v2 = np.array([logpdf_f(X[ii, :]) for ii in xrange(N)])
    print 'err2 %f' % np.log10(np.max(np.abs(v0 - v2)))


def test_mvn(runs=100):
    err = [0.0, 0.0, 0.0]
    for rr in xrange(runs):
        N = np.random.randint(1, 5)
        D = np.random.randint(1, 5)

        X = np.random.randn(N, D)
        mu = np.random.randn(D)
        S = np.random.randn(D, D)
        S = np.dot(S, S.T)

        logpdf0 = ss.multivariate_normal.logpdf(X, mu, S)

        inv_chol_U_cov = np.linalg.inv(np.linalg.cholesky(S).T)

        logpdf1 = p2.mvn_logpdf_from_chol(X, mu, inv_chol_U_cov)
        err[0] = max(err[0], np.max(np.abs(logpdf0 - logpdf1)))
        assert(np.allclose(logpdf0, logpdf1))

        X_tt = T.matrix()
        logpdf_tt = p3.mvn_logpdf_from_chol(X_tt, mu, inv_chol_U_cov)
        logpdf_f = theano.function([X_tt], logpdf_tt)
        logpdf2 = logpdf_f(X)
        err[1] = max(err[1], np.max(np.abs(logpdf0 - logpdf2)))
        assert(np.allclose(logpdf0, logpdf2))

        err[2] = max(err[2], np.max(np.abs(logpdf1 - logpdf2)))
    print 'mvn err P2 %f' % np.log10(err[0])
    print 'mvn err P3 %f' % np.log10(err[1])
    print 'mvn err P2-P3 %f' % np.log10(err[2])

np.random.seed(8525)

# TODO go in config
input_path = './local/phase2'
# TODO take as arg
N = 10

# Also test our mvn implementations while we are at it
test_mvn()

params_file_list = sorted(os.listdir(input_path))
for params_file in params_file_list:
    test_model(input_path, params_file, N)
print 'done'
