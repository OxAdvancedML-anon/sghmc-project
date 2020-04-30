import torch
import numpy as np
from torch import normal
from math import sqrt


class GenericUpdater:
    class LayerParams:
        def __init__(self, weights, momentums, regularisation_param=0.0):
            self.weights = weights
            self.momentums = momentums
            self.regularisation_param = regularisation_param

    def __init__(self, device, lr, momentum_coeff, num_train, noise, precision_params):
        self.device = device
        self.lr = lr
        self.momentum_coeff = momentum_coeff
        self.num_train = num_train
        self.noise = noise
        self.precision_params = precision_params
        self.gibbs_step = precision_params is not None
        self.layer_params_list = []

    def set_weights(self, layer_weights_list):
        for weights in layer_weights_list:
            self.layer_params_list.append(self.LayerParams(weights, torch.zeros_like(weights)))

    def zero_grad(self):
        for layer_params in self.layer_params_list:
            grad = layer_params.weights.grad
            if grad is not None:
                grad.detach_()
                grad.zero_()

    def run(self):
        for layer_params in self.layer_params_list:
            self.apply_update_rule(layer_params.weights.data, layer_params.momentums, layer_params.weights.grad.data,
                                   self.lr, self.momentum_coeff, layer_params.regularisation_param, self.num_train, self.noise)

    def run_gibbs_step(self):
        for layer_params in self.layer_params_list:
            with torch.no_grad():
                sampled_regularisation_param =\
                    self.apply_gibbs_rule(layer_params.weights, *self.precision_params) / self.num_train
            layer_params.regularisation_param = sampled_regularisation_param

    # Applies equation 15 (see paper) to update the parameters. Note that B in the original equation is assumed to be 0.
    def apply_update_rule(self, theta, v, grad, eta, alpha, lmbda, num_train, noise):
        delta_v = -eta * (grad + lmbda * theta) - alpha * v  # see eq 15.
        if noise:
            delta_v += normal(0.0, torch.empty(theta.shape).to(self.device)
                              .fill_(sqrt(2 * alpha * eta / num_train)))  # see eq 15.
        v.add_(delta_v)
        delta_theta = v  # see eq 15.
        theta.add_(delta_theta)

    def apply_gibbs_rule(self, theta, alpha, beta):
        param_count = np.prod(theta.shape)
        alpha += param_count / 2
        beta += torch.sum(torch.mul(theta, theta)).item() / 2
        lmbda = np.random.gamma(alpha, 1.0 / beta)
        return lmbda

class SGD(GenericUpdater):
    def __init__(self, device, lr, num_train):
        super().__init__(device, lr, momentum_coeff=1.0, num_train=num_train, noise=False, precision_params=None)


class SGDMomentum(GenericUpdater):
    def __init__(self, device, lr, momentum_coeff, num_train):
        super().__init__(device, lr, momentum_coeff, num_train=num_train, noise=False, precision_params=None)


class SGLD(GenericUpdater):
    def __init__(self, device, lr, num_train, precision_params):
        super().__init__(device, lr, momentum_coeff=1.0, num_train=num_train, noise=True, precision_params=precision_params)


class SGHMC(GenericUpdater):
    def __init__(self, device, lr, momentum_coeff, num_train, precision_params):
        super().__init__(device, lr, momentum_coeff, num_train=num_train, noise=True, precision_params=precision_params)
