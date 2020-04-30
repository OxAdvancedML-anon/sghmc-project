import numpy as np


class SGLD:
    """
    Stochastic Gradient Langevin Dynamics

    :param step_size: callable returning the step size / learning rate at a given time step t
    :param potential_grad: callable return the (noisy) gradient of the log posterior using a minibatch
    """
    def __init__(self, step_size, potential_grad):
        self.potential_grad = potential_grad
        self.step_size = step_size
        self.t = 0

    # resample_momentum is a dummy arg
    def sample(self, params, resample_momentum=True):
        batch_grad = self.potential_grad(params)

        eps = self.step_size(self.t)
        eta = np.random.normal(0, np.sqrt(eps), batch_grad.size)

        delta = -0.5 * eps * batch_grad + eta
        self.t += 1

        # Important that we don't use in-place sum here, generate a new sample instead
        return params + delta

    def reset(self):
        self.t = 0

