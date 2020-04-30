"""
Reproduces the 'simple demonstration' in the paper Bayesian Learning via
Stochastic Gradient Langevin Dynamics
"""
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

from algorithms import SGLD

var_1 = 10
var_2 = 1
var_x = 2
cov = np.array([[var_1, 0], [0, var_2]])
invcov = np.linalg.inv(cov)


def gen_data(n=100):
    x = 0.5 * np.sqrt(2) * np.random.randn(n) + \
        0.5 * (np.sqrt(2) * np.random.randn(n) + 1)
    return x


def grad_log_prior(theta):
    return np.log(2 * np.pi * np.sqrt(var_1 * var_2)) * 0.5 * np.matmul(invcov, theta)


def grad_log_likelihood(theta, x):
    sd_x = np.sqrt(var_x)
    exp_1 = np.exp(-0.5 * (x-theta[0])**2 / var_x)
    exp_2 = np.exp(-0.5 * (x-np.sum(theta))**2 / var_x)
    factor = 1 / (2 * sd_x * np.sqrt(np.pi * 2))
    
    grad = (1 / (exp_1 + exp_2)) * (1 / (2 * var_x)) * np.array(
        [(x - theta[0]) * exp_1 + (x-np.sum(theta)) * exp_2,
        ((x-np.sum(theta)) * exp_2)])
    return np.mean(grad, axis=1)


if __name__ == "__main__":
    runs = 1000
    x = gen_data()
    it = runs * len(x)

    theta = np.zeros(2)
    thetas = np.zeros((it, 2))

    gamma = 0.55
    b = it / (np.exp((-1/gamma) * np.log(0.01)) - 1)
    a = 0.01 / b ** (-gamma)

    eps_t = lambda t: a * (b + t) ** (-gamma)

    sgld = SGLD(grad_log_prior=grad_log_prior, grad_log_likelihood=grad_log_likelihood,
                step_size=eps_t, data=x, batch_size=1)

    for i in tqdm(range(it)):
        theta = sgld.sample(theta)
        thetas[i] = theta

    sns.jointplot(thetas[:, 0], thetas[:, 1], kind='kde')
    plt.show()
