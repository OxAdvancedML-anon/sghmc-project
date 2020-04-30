import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from math import floor, ceil, sqrt
from tqdm import tqdm

import algorithms
from algorithms import *



# Unit normal
def normal():
    log_volume = 0.91893853320467274178
    def val(x):
        return 0.5*x**2

    def grad(x):
        return x
    return val, grad, log_volume

# Himmelblau's function:
def paper():
    log_volume = 1.67992624289378653348
    def val(x):
        return -2*x**2 + x**4

    def grad(x):
        return -4*x + 4*x**3
    return val, grad, log_volume


def run_simulation(sampler, initial_params, sample_count, resample=50, burnin=100):

    param = initial_params
    samples = []

    np.seterr(all='raise')

    force_resample_next = False
    for i in tqdm(range(sample_count)):
        do_resample = i % resample == 0 or force_resample_next
        force_resample_next = False
        try:
            param = sampler.sample(param, resample_momentum=do_resample)
            if i >= burnin:
                samples.append(param)
        except ArithmeticError:
            print('Math error occured, param={}'.format(param))
            param = initial_params
            force_resample_next = True
            # remove samples leading up to divergence
            # the distribution is broken already anyway
            samples = samples[:max(0, len(samples)-resample)]

    if hasattr(sampler, 'correction') and sampler.correction:
        print('Metropolis-Hastings Acceptance Ratio = {:f}'.
            format(sampler.correction.accept_ratio()))
    return np.asarray(samples)


def plot_distribution(samples, grid, label=None):
    ys, xs = np.histogram(samples, bins=grid, density=True)
    xs = xs[:-1] + (xs[1]-xs[0])/2
    plt.plot(xs, ys, label=label, alpha=0.4)

def eval_divergence_sqrt(samples, div_est, start=100):
    points = []
    # keep bin size constant, otherwise (esp. for higher dimensions) large jumps
    # in bin count cause large universal shifts in entropy estimates
    bins = sqrt(len(samples))
    lb = 0
    for i in range(ceil(sqrt(start)), floor(sqrt(len(samples)))):
        ub = i**2
        points.append((ub, div_est.incremental_kld(samples[:ub], samples[lb:ub], bins)))
        lb = ub
    ub = len(samples)
    points.append((ub, div_est.incremental_kld(samples[:ub], samples[lb:ub], bins)))
    return zip(*points)

def print_div_stats(s, label, div_est):
    ce = div_est.cross_entropy(s)
    kld = div_est.kld(s)
    print('KL Div. + C = {:.4f}, C.E. + C = {:.4f}, Entropy = {:.4f} | {}'.format(kld, ce, ce - kld, label))

counter = 0
def save_figure(id_string, dpi=600):
    global counter
    dir = 'synthetic/{}'.format(id_string)
    os.makedirs(dir, exist_ok=True)
    path = '{}/{}.png'.format(dir, counter)
    plt.savefig(fname=path, bbox_inches='tight', dpi=dpi, transparent=True)
    counter += 1


def main():

    parser = argparse.ArgumentParser(description='Synthetic benchmark')
    parser.add_argument('-n', default=50000, type=int,
        help='Number of samples to generate', action='store')
    parser.add_argument('--dist', default=1, choices=range(2), type=int,
        help='Select test distribution (Unit Normal, Twin Peaks)', action='store')
    parser.add_argument('--noise', type=float,
        help='Overwrite default noise', action='store')
    parser.add_argument('--resample', type=int, default=50,
        help='Overwrite default momentum resampling period', action='store')
    parser.add_argument('-B', type=str,
        help='List of noise estimates for SGHMC', action='store')
    parser.add_argument('-C', type=str,
        help='List of friction terms for SGHMC (delta above B)', action='store')
    parser.add_argument('--entropy', help='Run entropy estimator tests', action='store_true')
    parser.add_argument('--true-normal', help='Run an analytic unit normal sampler', action='store_true')
    parser.add_argument('--cache', help='Load samples from cache instead', action='store_true')

    args = parser.parse_args()

    N = args.n
    switch = args.dist
    do_samples_from_unit_normal = args.dist == 0 and args.true_normal

    if args.entropy:
        run_entropy_tests(True)

    funs = [normal, paper]
    potential_energy, potential_grad, log_volume = funs[switch]()
    noise = [4.0, 4.0][switch] if args.noise is None else args.noise


    noise_sqrt = sqrt(noise)
    def stochastic_grad(x):
        return potential_grad(x) + np.random.normal(0, noise_sqrt)


    grid_res = 100
    grid = np.linspace(-2.0, 2.0, grid_res)


    initial_params=np.array([1.0])
    mass = np.array([[1.0]])
    step_size = 0.1
    step_count = 10

    true_Bhat = step_size * noise / 2
    Bhats = [0.0, true_Bhat] if args.B is None else [float(v) for v in args.B.split(',')]
    Cs = [] if args.C is None else [float(v) for v in args.C.split(',')]

    # copied from sgld_figures
    gamma = 0.55
    b = N / (np.exp((-1/gamma) * np.log(0.1)) - 1)
    a = step_size / b ** (-gamma)
    eps_t = lambda t: a * (b + t) ** (-gamma)

    def make_sghmc(C, Bhat, R=args.resample):
        return ('SGHMC$(C={},\\^B={})$ + $\\mathcal{{N}}(0,{})$'.format(C, Bhat, noise),
                SGHMC(stochastic_grad, mass, step_size, step_count,
                      C * np.identity(1), lambda x: Bhat * np.identity(1),
                      guaranteed_diagonal=True),
                R)

    enabled = [2, 3, 6, 7]

    id_string = 'Dist={},N={},V={},R={},Bs={},Cs={}'.format(switch, N, noise, args.resample, Bhats, Cs)

    samplers = [
        ('No MH',
         HMC(potential_grad, mass, step_size, step_count, None),
         args.resample),
        ('With MH',
         HMC(potential_grad, mass, step_size, step_count, potential_energy),
         args.resample),
        ('No MH + $\\mathcal{{N}}(0,{})$'.format(noise),
         HMC(stochastic_grad, mass, step_size, step_count, None),
         1),    # necessary to stop continual divergence
        ('With MH + $\\mathcal{{N}}(0,{})$'.format(noise),
         HMC(stochastic_grad, mass, step_size, step_count, potential_energy),
         args.resample),
        ('With MH (R=1) + $\\mathcal{{N}}(0,{})$'.format(noise),
         HMC(stochastic_grad, mass, step_size, step_count, potential_energy),
         1),
        ('SGLD + $\\mathcal{{N}}(0,{})$'.format(noise),
         SGLD(eps_t, stochastic_grad),
         args.resample),
        make_sghmc(true_Bhat, true_Bhat, args.resample),    # optimal settings
        make_sghmc(noise/2, 0, args.resample),              # lucky guess with unknown noise
    ]
    samplers = list(np.asarray(samplers)[enabled])
    samplers.extend([make_sghmc(c+b, b) for b in Bhats for c in Cs])


    if args.cache:
        print('Loading samples generated last time')
        all_samples = np.load('cache.npy', allow_pickle=True)
    else:
        all_samples = [(l, run_simulation(s, initial_params, N, rs)) for l, s, rs in samplers]

    if not args.cache:
        np.save('cache.npy', all_samples)

    if do_samples_from_unit_normal:
        samples_true = np.random.normal(0, 1, N)
        all_samples.append(('Analytic $\\mathcal{{N}}(0,1)$', samples_true))

    # can use analytic results for simple functions (pre-set)
    # log_volume = np.log(integrate_volume_1d(potential_energy, samples_mh))
    print('Log volume = {}'.format(log_volume))


    # for these functions we can normalise the pdf,
    # so we obtain the 'actual' KL-divergence (barring the entropy bias)
    plt.figure('KL Divergence', figsize=(16,9))

    for l, s in all_samples:
        idx, klds = eval_divergence_sqrt(s,
            DistDivergence(lambda x: -potential_energy(x)-log_volume))
        np.seterr(all='ignore')   # bug in MPL, otherwise throws a FPE
        plt.semilogy(idx, klds, label=l)

    plt.legend()

    save_figure(id_string)

    plt.figure('Probability Density', figsize=(16,9))

    for l, s in all_samples:
        plot_distribution(s, grid, label=l)
        # print_entropy_table(s, label=l)

    # Generates the 'true' plot for the distribution
    plt.plot(grid, np.exp(-potential_energy(grid)-log_volume), 'k:', label='True Distribution')

    plt.legend()

    save_figure(id_string)

    div_est = DistDivergence(lambda x: -potential_energy(x)-log_volume)
    for l, s in all_samples:
        print_div_stats(s, l, div_est)

    plt.show()


if __name__ == "__main__":
    main()
