import numpy as np
from numpy.linalg import inv
import scipy.stats

class SGHMC:
    """
        Stochastic Gradient Hamiltonian Monte Carlo (no Metropolis-Hastings correction)
        :param grad_log_prior: callable returning the gradient of the log prior [p(theta_t)]
        :param grad_log_likelihood: callable returning the gradient of the likelihood of a sample
            [p(x_t_i | theta_i)]
        :param data: full dataset to sample from
        :param batch_size: size of mini batches to use in each step
        :param mass: 2-D array containing the mass for Hamiltonian dynamics
            (M in the paper)
        :param step_size: the size of a step taken by the algorithm (epsilon
            in the paper)
        :param step_count: the number of steps to take in the algorithm (m in the
            paper)
        :param friction_term: A user provided friction term to counteract gradient (C in the paper)
        :param noise_model_estimate: A callable returning an estimate of the noise at a given theta (B hat in the paper)
        """

    def __init__(self, potential_grad, mass, step_size, step_count, friction_term, noise_model_estimate = None, guaranteed_diagonal=False):
        self.potential_grad = potential_grad
        self.mass = mass
        self.inv_mass = inv(mass)
        self.step_size = step_size
        self.num_steps = step_count
        self._C = friction_term
        self.momentum = scipy.stats.multivariate_normal.rvs(mean=None, cov=self.mass)
        self._Bhat = noise_model_estimate or self.ZERO
        self.diag_mass = self.check_diag(self.mass)
        if guaranteed_diagonal:
            self.diag_noise = True
        elif noise_model_estimate is None:
            self.diag_noise = self.check_diag(self._C)
        else:
            self.diag_noise = False
        self.t = 0

    def sample(self, params, resample_momentum=True):
        # 'params' is the vector of 'positions' in the monte carlo algorithm
        # (theta in the paper)
        # 'momentum' is the set of auxiliary momentum variables (r in the paper)
        theta = params
        momentum = self.generate_noise(mean=np.zeros(theta.size), cov=self.mass, known_diag=self.diag_mass) if resample_momentum else self.momentum
        step_size_t = self.step_size(self.t) if callable(self.step_size) else self.step_size
        self.t += 1

        for _ in range(self.num_steps):
            theta = theta + step_size_t * self.inv_mass.dot(momentum)

            # Add gradient descent + friction update
            momentum = momentum - step_size_t * (self.potential_grad(theta) + self._C.dot(self.inv_mass.dot(momentum)))

            # Inject noise
            momentum = momentum + self.generate_noise(mean=np.zeros(theta.size), cov=2*step_size_t*(self._C - self._Bhat(theta)), known_diag=self.diag_noise)

        self.momentum = momentum
        return theta

    def generate_noise(self, mean, cov, known_diag=False):
        diagonal = known_diag or cov.size==1 or self.check_diag(cov)
        if diagonal:
            return np.random.normal(mean, np.sqrt(np.diagonal(cov)), mean.size)
        else:
            return np.random.multivariate_normal(mean, cov)

    def check_diag(self, cov):
        i, j = cov.shape
        test = cov.reshape(-1)[:-1].reshape(i-1, j+1)
        return ~np.any(test[:, 1:])

    @staticmethod
    def ZERO(theta):
        return 0
