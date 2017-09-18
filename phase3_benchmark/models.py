# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import theano.tensor as T

from theano.sandbox.linalg.ops import Det, MatrixInverse, Cholesky

det = Det()
minv = MatrixInverse()
chol = Cholesky()


def mvn_logpdf(sample, mean, cov):
    """Return a theano expression representing the values of the log prob
    density function of the multivariate normal.
    Parameters
    ----------
    sample : Theano variable
        Array of shape ``(n, d)`` where ``n`` is the number of samples and
        ``d`` the dimensionality of the data.
    mean : Theano variable
        Array of shape ``(d,)`` representing the mean of the distribution.
    cov : Theano variable
        Array of shape ``(d, d)`` representing the covariance of the
        distribution.
    Returns
    -------
    l : Theano variable
        Array of shape ``(n,)`` where each entry represents the log density of
        the corresponding sample.
    Examples
    --------
    >>> import theano
    >>> import theano.tensor as T
    >>> import numpy as np
    >>> from breze.learn.utils import theano_floatx
    >>> sample = T.matrix('sample')
    >>> mean = T.vector('mean')
    >>> cov = T.matrix('cov')
    >>> p = logpdf(sample, mean, cov)
    >>> f_p = theano.function([sample, mean, cov], p)
    >>> mu = np.array([-1, 1])
    >>> sigma = np.array([[.9, .4], [.4, .3]])
    >>> X = np.array([[-1, 1], [1, -1]])
    >>> mu, sigma, X = theano_floatx(mu, sigma, X)
    >>> ps = f_p(X, mu, sigma)
    >>> np.allclose(ps, np.log([4.798702e-01, 7.73744047e-17]))
    True
    """
    # TODO replace with cholesky stuff

    inv_cov = minv(cov)

    inv_cov = minv(cov)
    L = chol(inv_cov)

    log_part_func = (
        - .5 * T.log(det(cov))
        - .5 * sample.shape[1] * T.log(2 * np.pi))

    mean = T.shape_padleft(mean)
    residual = sample - mean
    B = T.dot(residual, L)
    A = (B ** 2).sum(axis=1)
    log_density = - .5 * A

    return log_density + log_part_func


def logsumexp_tt(X, axis=None):
    # Supposedly theano is smart enough to convert this
    Y = T.log(T.sum(T.exp(X), axis=axis))
    return Y


def outer_tt(x, y):
    '''T.outer() does not work when y is np array.'''
    return T.dot(T.shape_padright(x), T.shape_padleft(y))


def MoG(x, params):
    # TODO x is theano vector?? check.
    assert(params['type'] == 'full')

    w = params['weights']
    w = w / np.sum(w)  # Just to be sure normalized
    n_mixtures = len(w)

    # TODO use scan
    loglik_mix = [None] * n_mixtures
    for mm in xrange(n_mixtures):
        # Could check posdef if we wanted to be extra safe
        mu, S = params['means'][mm, :], params['covariances'][mm, :, :]
        # TODO use padleft or right
        loglik_mix[mm] = mvn_logpdf(T.shape_padleft(x), mu, S)[0]
    loglik_mix_T = T.log(w) + T.stack(loglik_mix, axis=0)
    # Way to force theano to use logsumexp??
    logpdf = logsumexp_tt(loglik_mix_T, axis=0)
    return logpdf


def _MoG_sample(w, mus, covs, N=1):
    # TODO test against sklearn version
    D = mus.shape[1]
    n_mixtures = len(w)

    k = np.random.choice(n_mixtures, N=N, replace=True, p=w)

    X = np.zeros((N, D))
    for nn, mm in enumerate(k):
        # Note: This is not an efficient way to do it
        mu, S = mus[mm, :], covs[mm, :, :]
        X[nn, :] = np.random.multivariate_normal(mu, S)
    return X

def MoG_sample(params, N=1):
    assert(params['type'] == 'full')
    X = _MoG_sample(params['weights'], params['means'], params['covariances'],
                    N=N)
    return X


# TODO resolve dir struct to re-use IGN code
'''
def IGN(x, params):
    # TODO x is theano vector?? check.
    base_logpdf = t_util.norm_logpdf_T if params['gauss_basepdf'] \
        else t_util.t_logpdf_T

    layers = dict(params)
    # ign_log_pdf() checks if there are any extra elements in param dict
    del layers['gauss_basepdf']
    logpdf, _ = ign.ign_log_pdf_T(T.shape_padleft(x), layers, base_logpdf)
    logpdf_ = logpdf[0]  # Unpack extra dimension
    return logpdf_
'''


def RNADE(x, params):
    assert(x.ndim == 1)  # Assuming x is theano vector here
    x = T.shape_padleft(x)  # 1 x V

    # TODO infer these from parameters
    n_hidden, n_layers = params['n_hidden'], params['n_layers']

    # TODO document/check all dims
    Wflags, W1, b1 = params['Wflags'], params['W1'], params['b1']
    Ws, bs = params['Ws'], params['bs']
    V_alpha, b_alpha = params['V_alpha'], params['b_alpha']
    V_mu, b_mu = params['V_mu'], params['b_mu']
    V_sigma, b_sigma = params['V_sigma'], params['b_sigma']
    orderings = params['orderings']
    D = len(orderings[0])

    assert(params['nonlinearity'] == 'RLU')  # Only one supported yet
    act_fun = T.nnet.relu
    softmax = T.nnet.softmax
    pl = T.shape_padleft
    pr = T.shape_padright

    N = x.shape[0]  # Should be 1 for now.

    lp = []
    for o_index, curr_order in enumerate(orderings):
        assert(len(curr_order) == D)

        a = T.zeros((N, n_hidden)) + pl(b1)  # N x H
        lp_curr = []
        for i in curr_order:
            h = act_fun(a)  # N x H
            for l in xrange(n_layers - 1):
                h = act_fun(T.dot(h, Ws[l, :, :]) + pl(bs[l, :]))  # N x H

            # All N x C
            z_alpha = T.dot(h, V_alpha[i, :, :]) + pl(b_alpha[i, :])
            z_mu = T.dot(h, V_mu[i, :, :]) + pl(b_mu[i, :])
            z_sigma = T.dot(h, V_sigma[i, :, :]) + pl(b_sigma[i, :])

            # Any final warping. All N x C.
            # Alpha = T.exp(z_alpha) / T.sum(T.exp(z_alpha), axis=1, keepdims=True)
            Alpha = softmax(z_alpha)
            Mu = z_mu
            Sigma = T.exp(z_sigma)  # TODO be explicit this is std

            lp_components = -0.5 * ((Mu - pr(x[:, i])) / Sigma) ** 2 \
                - T.log(Sigma) - 0.5 * T.log(2 * np.pi) + T.log(Alpha)  # N x C
            lpc = logsumexp_tt(lp_components, axis=1)
            lp_curr.append(lpc)

            bias = pl(Wflags[i, :])
            prod = outer_tt(x[:, i], W1[i, :])  # pl(x[0, i] * W1[i, 0])
            update = prod + bias
            a = a + update  # N x H
        # The following only works with N=1!!
        # lp.append(T.sum(lp_curr, axis=1) + T.log(1.0 / len(orderings)))
        lp.append(T.sum(lp_curr) + T.log(1.0 / len(orderings)))
    logpdf = logsumexp_tt(lp, axis=0)
    return logpdf


def _RNADE_sample(params):
    N = 1  # TODO generalize

    n_hidden, n_layers = params['n_hidden'], params['n_layers']

    Wflags, W1, b1 = params['Wflags'], params['W1'], params['b1']
    Ws, bs = params['Ws'], params['bs']
    V_alpha, b_alpha = params['V_alpha'], params['b_alpha']
    V_mu, b_mu = params['V_mu'], params['b_mu']
    V_sigma, b_sigma = params['V_sigma'], params['b_sigma']
    orderings = params['orderings']

    assert(params['nonlinearity'] == 'RLU')  # Only one supported yet
    act_fun = lambda x_: x_ * (x_ > 0.0)

    def softmax(X):
        '''Calculates softmax row-wise'''
        # TODO implement logsoftmax
        # TODO move to util
        X = X - np.max(X, axis=1, keepdims=True)
        e = np.exp(X)
        R = e / np.sum(e, axis=1, keepdims=True)
        return R

    o_index = np.random.choice(len(orderings))
    order_used = orderings[o_index]

    X = np.zeros((N, len(order_used)))
    a = np.zeros((N, n_hidden)) + b1[None, :]  # N x H
    for i in order_used:
        h = act_fun(a)  # N x H
        for l in xrange(n_layers - 1):
            h = act_fun(np.dot(h, Ws[l, :, :]) + bs[l, None])  # N x H

            # All N x C
            z_alpha = np.dot(h, V_alpha[i, :, :]) + b_alpha[i, None]
            z_mu = np.dot(h, V_mu[i, :, :]) + b_mu[i, None]
            z_sigma = np.dot(h, V_sigma[i, :, :]) + b_sigma[i, None]

            # Any final warping. All N x C.
            Alpha = softmax(z_alpha)
            Mu = z_mu
            Sigma = np.exp(z_sigma)  # TODO be explicit this is std

            # TODO generalize to N > 1, move to subroutine
            k = np.random.choice(Alpha.shape[1], p=Alpha[0, :])
            X[0, i] = Mu[0, k] + Sigma[0, k] * np.random.randn()

            a += np.outer(X[:, i], W1[i, :]) + Wflags[i, None]  # N x H
    return X


def RNADE_sample(params, N=1):
    X = np.concatenate([_RNADE_sample(params) for _ in xrange(N)], axis=0)
    return X

BUILD_MODEL = {'MoG': MoG, 'RNADE': RNADE}
SAMPLE_MODEL = {'MoG': MoG_sample, 'RNADE': RNADE_sample}
