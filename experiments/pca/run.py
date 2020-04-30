import math
import numpy as np
import scipy as sp
from algorithms import HMC
from tqdm import tqdm

from experiments.pca.helpers import vectorize_params, unvectorize_params
from experiments.pca.bpca import BPCA


if __name__ == '__main__':
    N = 30
    # hyperparameters
    shape_alpha = 0.7 # keeps this wide
    scale_alpha = 1.0
    beta = 1.0
    shape_tau = 0.7
    scale_tau = 2.0

    # N, d would be derived from dataset not other way round
    d = 10
    q = d-1

    # generate some starting parameters from the priors
    mu = np.random.normal(np.zeros(d), np.ones(d) / np.sqrt(beta))
    alpha = np.log(np.random.gamma(shape_alpha, scale_alpha, size=(q,)))

    W = np.random.multivariate_normal(
        np.zeros((d, q)).ravel(), 
        np.kron(np.identity(d), np.identity(q) * 1/np.exp(alpha))).reshape((d,q))
    tau = np.log(np.random.gamma(shape_tau, scale_tau))
    params = vectorize_params(W, mu, tau, alpha)

    cov = np.diag(np.array([10,6,6,3,1,1,1,1,1,1])**2)
    data = np.random.multivariate_normal(np.zeros(d), cov, size=N)

    # Init BPCA
    bpca = BPCA(shape_alpha=shape_alpha, scale_alpha=scale_alpha, beta=beta, shape_tau=shape_tau,
                scale_tau=scale_tau)
    bpca.setup(data, q)

    print('alpha log prior', bpca.log_prior_alpha(alpha))
    print('tau log prior', bpca.log_prior_tau(tau))
    print('W log prior', bpca.log_prior_W(W, alpha))
    print('mu log prior', bpca.log_prior_mu(mu))
    print('potential energy', bpca.potential_energy(data, W, mu, tau, alpha))

    hmc_sampler = HMC(potential_grad=bpca.vectorized_grad_U, mass=np.identity(params.shape[0]), 
        step_size=0.005, step_count=5)

    trace = bpca.run(sampler=hmc_sampler, iterations=5000, start_params=params)

    end_samples = list(map(lambda s: unvectorize_params(s, N, d, q), trace))
    alphas = np.array([np.exp(s[-1]) for s in end_samples[200:]])

    print(np.mean(alphas, axis=0))



