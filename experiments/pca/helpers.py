import numpy as np
from scipy import special


def loggamma_logpdf(x, shape, scale=1.0):
    return (shape * x - np.exp(x) / scale) - np.log(np.power(scale, shape)) - special.loggamma(shape)


def vectorize_params(W, mu, tau, alpha):
    return np.concatenate((
        W.ravel(), mu, np.array(tau).reshape(1,), alpha))


def unvectorize_params(params, N, d, q):
    W = params[:d*q].reshape(d,q)
    mu = params[d*q:d*q + d]
    tau = params[d*q +d]
    alpha = params[-q:]
    return W, mu, tau, alpha
