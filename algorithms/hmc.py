import numpy as np
from numpy.linalg import inv

from .mh_correction import MHCorrection

class HMC:
    """
        Hamiltonian Monte Carlo (no Metropolis-Hastings correction)

        :param potential_grad: callable returning the gradient of the potential
            energy (del U in the paper)
        :param mass: 2-D array containing the mass for Hamiltonian dynamics
            (M in the paper)
        :param step_size: the size of a step taken by the algorithm (epsilon
            in the paper)
        :param step_count: the number of steps to take in the algorithm (m in the
            paper)
        """

    def __init__(self, potential_grad, mass, step_size, step_count, potential_energy=None):
        self.potential_grad = potential_grad
        self.mass = mass
        self.inv_mass = inv(mass)
        self.step_size = step_size
        assert(step_count >= 1)
        self.num_steps = step_count

        self.momentum = None
        if potential_energy:
            self.correction = MHCorrection(potential_energy, lambda x: self._kinetic_energy(x))
        else: self.correction = None

        # Check if the mass is diagonal so we can speed up sampling later
        # https://stackoverflow.com/a/43885215
        i, j = mass.shape
        test = mass.reshape(-1)[:-1].reshape(i-1, j+1)
        self.diagonal_mass = ~np.any(test[:, 1:])


    def _kinetic_energy(self, momentum):
        return 0.5 * momentum.T @ self.inv_mass @ momentum

    def sample(self, params, resample_momentum=True):
        # 'params' is the vector of 'positions' in the monte carlo algorithm
        # (theta in the paper)
        # 'momentum' is the set of auxiliary momentum variables (r in the paper)
        params = np.asarray(params)
        if (not resample_momentum) and self.momentum is not None:
            momentum = self.momentum
        elif self.diagonal_mass:
            momentum = np.random.normal(np.zeros(params.size), np.sqrt(np.diagonal(self.mass)), params.size)
        else:
            momentum = np.random.multivariate_normal(np.zeros(params.size), self.mass)

        if self.correction:
            params0 = np.copy(params)
            momentum0 = np.copy(momentum)

        momentum -= (self.step_size / 2) * self.potential_grad(params)

        for _ in range(self.num_steps - 1):
            params = params + self.step_size * self.inv_mass @ momentum
            momentum = momentum - self.step_size * self.potential_grad(params)

        params = params + self.step_size * self.inv_mass @ momentum
        momentum -= (self.step_size / 2) * self.potential_grad(params)

        # !!! paper incorrectly overshoots momentum by one eps,
        # which distorts energy conservation and makes MH
        # try to correct towards a distorted distribution !!!

        if self.correction is not None and\
            not self.correction((params0, momentum0), (params, momentum)):
                # if a sample is rejected we cannot preserve the computed
                # momentum, otherwise MH behaves extemely wrong
                # this is probably because we break independence between
                # kinetic and potential energy, so the marginal distribution
                # of potential energy no longer corresponds to the target dist
                self.momentum = momentum0
                return params0

        # Store momentum in case we do not resample
        self.momentum = momentum
        return params
