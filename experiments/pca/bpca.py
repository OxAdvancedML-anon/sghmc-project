import numpy as np
from scipy import stats, special
from tqdm import tqdm

from experiments.pca.helpers import vectorize_params, unvectorize_params, loggamma_logpdf

class BPCA:
    data = None
    N = None
    d = None
    q = None

    def __init__(self, shape_alpha, scale_alpha, shape_tau, scale_tau, beta):
        self.shape_alpha = shape_alpha
        self.scale_alpha = scale_alpha
        self.shape_tau = shape_tau
        self.scale_tau = scale_tau
        self.beta = beta

    def setup(self, data, q):
        self.data = data
        self.N = data.shape[0]
        self.d = data.shape[1]
        self.q = q
        self.t = 0

    def reset_sampler(self, batch_size):
        self.batch_size = batch_size
        self.t = 0
        
    def run(self, sampler, start_params, iterations, **kwargs):
        params = start_params
        trace = [params]
        for i in tqdm(range(iterations)):
            new_params = sampler.sample(params, **kwargs)
            self.t += 1

            if np.all(np.isclose(new_params, params)):
                # Repeat step until we get an accepted sample so we
                # end up with them same number of samples for fairness
                repeats = 0
                while np.all(np.isclose(new_params, params)):
                    new_params = sampler.sample(params)
                    repeats += 1
            trace.append(new_params)
            params = new_params
        return trace

    def log_prior_mu(self, mu):
        """
        mu ~ N(0, beta^-1 * I_d)
        """
        return stats.multivariate_normal.logpdf(
            mu, mean=np.zeros(self.d), cov=np.identity(self.d) / self.beta)

    def log_prior_alpha(self, alpha):
        """
        alpha ~ LogGamma(shape_alpha, scale_alpha)
        alpha here is actually log(alpha) to extend support. Other functions here
        use exp(alpha) compared to the paper
        """
        return np.sum([loggamma_logpdf(a, self.shape_alpha, scale=self.scale_alpha) 
            for a in alpha])

    # W ~ as in paper (matrix variate normal?)
    # from paper but with exp(alpha) instead of alpha
    def log_prior_W(self, W, alpha):
        """
        W ~ matrix normal distribution given alpha
        """
        return np.sum([
            np.log((np.exp(alpha[i]) / 2 * np.pi) ** (self.d / 2.0)) + \
                (-0.5 * np.exp(alpha[i]) * (W[:, i].T @ W[:, i]))
            for i in range(self.q)])

    def log_prior_tau(self, tau):
        """
        tau ~ Log-Gamma(shape_tau, scale_tau)
        Same log transformation as alpha
        """
        return loggamma_logpdf(tau, self.shape_tau, scale=self.scale_tau)

    def log_p_t(self, t, W, mu, tau, alpha):
        """
        p(t) = N(mu, WW^T + sigma^2 I_d)
        Note this now isn't used since the LL function below is more optimised
        """
        return stats.multivariate_normal.logpdf(t, mean=mu, cov=(W @ W.T) + np.identity(self.d) / np.exp(tau))

    def log_prior(self, W, mu, tau, alpha):
        return self.log_prior_mu(mu) + self.log_prior_alpha(alpha) + \
            self.log_prior_W(W, alpha) + self.log_prior_tau(tau)

    def log_likelihood(self, D, W, mu, tau, alpha):
        """
        Faster version than summing log_p_t: 
            doesn't repeatedly calculate covariance inverse
        """
        C = (W @ W.T) + np.identity(self.d) / np.exp(tau)
        C_inv = np.linalg.inv(C)
        N = D.shape[0]
        def f(t):
            v = t - mu
            return -1/2 * (v.T @ C_inv @ v)
        ll = -N/2 * (self.d * np.log(2*np.pi) + np.log(np.linalg.det(C)))
        ll += np.sum([f(D[i]) for i in range(N)])
        return ll

    def potential_energy(self, D, W, mu, tau, alpha):
        return -self.log_likelihood(D, W, mu, tau, alpha) - self.log_prior(W, mu, tau, alpha)

    # Various forms of this function for different sampling algorithms and testing situations
    def vectorized_U(self, params):
        if self.data is None:
            raise Exception('Run setup(data, q) before using U')

        W, mu, tau, alpha = unvectorize_params(params, self.N, self.d, self.q)
        return self.potential_energy(self.data, W, mu, tau, alpha)
    
    def vectorized_grad_U(self, params):
        """
        Includes the scaling factor - used for SGHMC. Should probably clean this up once
        everything is merged. 
        """
        if self.data is None:
            raise Exception('Run setup(data, q) before using grad_U')
        W, mu, tau, alpha = unvectorize_params(params, self.N, self.d, self.q)

        idx = (self.t * self.batch_size)
        batch = self.data.take(list(range(idx, idx+self.batch_size)), mode='wrap', axis=0)

        scaling_factor = len(self.data) / len(batch)
        return - scaling_factor * self.vectorized_grad_ll(params, batch) - self.vectorized_grad_lp(params)

    def vectorized_grad_ll(self, params, batch):
        """
        Grad log-likelihood for a batch of data - unadjusted by the scaling factor
        """
        W, mu, tau, alpha = unvectorize_params(params, self.N, self.d, self.q)
        return vectorize_params(*self.grad_log_likelihood(batch, W, mu, tau, alpha))

    def vectorized_grad_lp(self, params):
        W, mu, tau, alpha = unvectorize_params(params, self.N, self.d, self.q)
        return vectorize_params(*self.grad_log_prior(W, mu, tau, alpha))

    # GRADIENTS
    def grad_log_likelihood(self, D, W, mu, tau, alpha):
        N = D.shape[0]
        S = 1/N * np.sum([
            np.outer(D[i] - mu, D[i] - mu) for i in range(N)],
            axis=0)
        C = (W @ W.T) + np.eye(self.d) / np.exp(tau)
        C_inv = np.linalg.inv(C)
        CSC = C_inv @ S @ C_inv  # used frequently
        grad_W = N * (CSC @ W - C_inv @ W)
        grad_mu = C_inv @ np.sum(D - mu, axis=0)
        grad_tau = (N / (2 * np.exp(tau))) * (np.trace(C_inv) - np.trace(CSC))
        return grad_W, grad_mu, grad_tau, np.zeros(self.q)

    def grad_log_prior(self, W, mu, tau, alpha):
        grad_W = - W * np.exp(alpha)  # - e^{a_j} * w_ij
        grad_mu = - self.beta * mu
        grad_tau = self.shape_tau - np.exp(tau) / self.scale_tau

        # parts from both W prior and alpha prior
        grad_alpha = (self.shape_alpha - np.exp(alpha) / self.scale_alpha + 
            0.5 * (self.d - np.exp(alpha) * np.diag(W.T @ W)))

        return grad_W, grad_mu, grad_tau, grad_alpha

    def grad_potential_energy(self, D, W, mu, tau, alpha):
        lp_grads = self.grad_log_prior(W, mu, tau, alpha)
        ll_grads = self.grad_log_likelihood(X, W, mu, tau, alpha)
        return [-ll_grads[i]-lp_grads[i] for i in range(5)]