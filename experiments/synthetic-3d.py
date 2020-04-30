import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib import cm
from matplotlib import colors as cl
from mpl_toolkits.mplot3d import Axes3D
from numpy import exp, log

from .synthetic import run_simulation, save_figure, eval_divergence_sqrt, print_div_stats
from algorithms import *



# Poor man's namespaces

# Squircle with hole
def squircle():
    a, b, c, d = 0.125, 0.5, 1, 0.5
    def val(x):
        return a*x[0]**4 + b*x[1]**4 + c/(x[0]**4 + x[1]**4 + d)

    def grad(x):
        return np.asarray([
            4*a*x[0]**3 - 4*c*x[0]**3 / (x[0]**4 + x[1]**4 + d)**2,
            4*b*x[1]**3 - 4*c*x[1]**3 / (x[0]**4 + x[1]**4 + d)**2])
    return val, grad

# Himmelblau's function:
def himmelblau():
    t = 0.01
    def val(x):
        return t*( (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2 )

    def grad(x):
        return t*np.asarray([
            4*(x[0]**2 + x[1] - 11)*x[0] + 2*(x[0] + x[1]**2 - 7),
            2*(x[0]**2 + x[1] - 11) + 4*(x[0] + x[1]**2 - 7)*x[1]])
    return val, grad

# Rosenbrock's function:
def rosenbrock():
    a, b, c, d = -13.0, 10.0, 10, -3.0
    t = 0.002
    def val(x):
        return t*( (a - x[0])**2 + b* (d*x[1] - x[0]**2) **2 -c*x[1] )

    def grad(x):
        return t*np.asarray([
            -2*(a - x[0]) - 4*b*(d*x[1] - x[0]**2)*x[0],
            2*b*(d*x[1] - x[0]**2)*d - c])
    return val, grad


resolutionA = 120
resolutionB = 32
cmapA = cm.cividis
cmapB = cm.plasma
gamma = 1


def plot_distribution_3d(samples, x, y, label=None):
    z, edges = np.histogramdd(samples, bins=(x, y), density=True)

    x = x[:-1] + (x[1]-x[0])/2
    y = y[:-1] + (y[1]-y[0])/2
    x, y = np.meshgrid(x, y, indexing='ij')

    fig = plt.figure(label)
    ax = fig.gca(projection='3d', adjustable='box', title=label)
    ax.plot_trisurf(x.ravel(), y.ravel(), z.ravel(),
        cmap=cmapB, antialiased=True)

def plot_distribution_heatmap(samples, label=None):
    xs, ys = zip(*samples)
    plt.figure(label + ' Heatmap (γ={})'.format(gamma))
    plt.title(label)
    plt.hist2d(xs, ys, bins=resolutionA, density=True, cmap=cmapA, norm=cl.PowerNorm(gamma))

def plot_function_3d(x, y, z, label=None):
    fig = plt.figure(label)
    ax = fig.gca(projection='3d', adjustable='box', title=label)
    ax.contourf(x, y, z, 256, cmap=cmapB, antialiased=True)

def plot_function_heatmap(x, y, z, label=None):
    plt.figure(label + ' Heatmap (γ={})'.format(gamma))
    plt.title(label)
    plt.contourf(x, y, z, 256, cmap=cmapA, norm=cl.PowerNorm(gamma))


def main():

    parser = argparse.ArgumentParser(description='Synthetic 3D benchmark')
    parser.add_argument('-n', default=50000, type=int,
        help='Number of samples to generate', action='store')
    parser.add_argument('--dist', default=1, choices=range(3), type=int,
        help='Select test distribution (Squircle, Himmelblau\'s, Rosenbrock\'s)', action='store')
    parser.add_argument('--noise', type=float,
        help='Overwrite default noise', action='store')
    parser.add_argument('--resample', type=int, default=50,
        help='Overwrite default momentum resampling period', action='store')
    parser.add_argument('-B', type=str,
        help='List of noise estimates for SGHMC', action='store')
    parser.add_argument('-C', type=str,
        help='List of friction terms for SGHMC (delta above B)', action='store')
    parser.add_argument('--entropy', help='Run entropy estimator tests', action='store_true')
    parser.add_argument('--no-heatmap', '-H', help='Do not show heatmaps', action='store_true')
    parser.add_argument('--no-surface', '-S', help='Do not show surface plots', action='store_true')
    parser.add_argument('--no-entropy', '-E', help='Do not show entropy tables', action='store_true')
    parser.add_argument('--gamma', default=1.0, type=float, help='Adjust heatmap gamma', action='store')
    parser.add_argument('--cache', help='Load samples from cache instead', action='store_true')

    args = parser.parse_args()

    N = args.n
    switch = args.dist
    show_heatmaps = not args.no_heatmap
    show_surfaces = not args.no_surface
    show_entropies = not args.no_entropy
    global gamma
    gamma = args.gamma

    if args.entropy:
        run_entropy_tests(True)

    funs = [squircle, himmelblau, rosenbrock]
    potential_energy, potential_grad = funs[switch]()
    # it seems the effect of uncancelled noise vanishes extremely quickly
    # as its magnitude decreases
    noise = [5.0, 20.0, 20.0][switch] if args.noise is None else args.noise

    noise_sqrt = sqrt(noise)
    def stochastic_grad(x):
        return potential_grad(x) +\
            [np.random.normal(0, noise_sqrt), np.random.normal(0, noise_sqrt)]


    initial_params=np.array([0.1, 0.1])
    mass = np.identity(2)
    step_size = 0.1
    step_count = 10

    true_Bhat = step_size * noise / 2
    Bhats = [0.0, true_Bhat] if args.B is None else [float(v) for v in args.B.split(',')]
    Cs = [] if args.C is None else [float(v) for v in args.C.split(',')]

    # copied from sgld_figures
    g = 0.55
    b = N / (np.exp((-1/g) * np.log(0.1)) - 1)
    a = step_size / b ** (-g)
    eps_t = lambda t: a * (b + t) ** (-g)

    def make_sghmc(C, Bhat, R=args.resample):
        return ('SGHMC$(C={},\\^B={})$ + $\\mathcal{{N}}(0,{})$'.format(C, Bhat, noise),
                SGHMC(stochastic_grad, mass, step_size, step_count,
                      C * np.identity(2), lambda x: Bhat * np.identity(2),
                      guaranteed_diagonal=True),
                R)

    enabled = [2, 3, 6, 7]

    id_string = '3D,Dist={},N={},V={},R={},Bs={},Cs={}'.format(switch, N, noise, args.resample, Bhats, Cs)

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
        all_samples = np.load('cache3d.npy', allow_pickle=True)
    else:
        all_samples = [(l, run_simulation(s, initial_params, N, rs)) for l, s, rs in samplers]

    if not args.cache:
        np.save('cache3d.npy', all_samples)

    # Get common domain for all plots
    just_samples = np.concatenate([s for l, s in all_samples])
    q_vals = np.array([-potential_energy(s) for s in just_samples])
    just_samples = just_samples[q_vals >= np.median(q_vals)]
    lbs = np.amin(just_samples, axis=0)
    ubs = np.amax(just_samples, axis=0)
    s = (ubs - lbs) / 4.0
    lbs -= s
    ubs += s
    xs, ys = xys = np.meshgrid(np.linspace(lbs[0], ubs[0], resolutionA),
                               np.linspace(lbs[1], ubs[1], resolutionA),
                               indexing='ij')

    # samples are passed just for estimating the domain
    log_volume = np.log(integrate_volume_2d(potential_energy, just_samples))
    print('Log volume = {}'.format(log_volume))

    plt.figure('KL Divergence + C', figsize=(16,9))

    for l, s in all_samples:
        idx, klds = eval_divergence_sqrt(s,
            DistDivergence(lambda x: -potential_energy(x)-log_volume),
            start=min(5000, N//4))
        # eps = 10**-8
        # klds = np.maximum(klds, eps)
        np.seterr(all='ignore')   # bug in MPL, otherwise throws a FPE
        plt.semilogx(idx, klds, label=l)

    plt.legend()

    save_figure(id_string)

    x_ax = np.linspace(lbs[0], ubs[0], resolutionB)
    y_ax = np.linspace(lbs[1], ubs[1], resolutionB)
    for l, s in all_samples:
        if show_surfaces:
            plot_distribution_3d(s, x_ax, y_ax, label=l)
            save_figure(id_string)
        if show_heatmaps:
            plot_distribution_heatmap(s, label=l)
            save_figure(id_string)
        if show_entropies:
            print_entropy_table(s, label=l)

    # Generates the 'true' plot for the distribution
    zs = np.exp(-potential_energy(xys)-log_volume)

    if show_surfaces:
        plot_function_3d(xs, ys, zs, label='True Distribution')
        save_figure(id_string)
    if show_heatmaps:
        plot_function_heatmap(xs, ys, zs, label='True Distribution')
        save_figure(id_string)


    div_est = DistDivergence(lambda x: -potential_energy(x)-log_volume)
    for l, s in all_samples:
        print_div_stats(s, l, div_est)

    plt.show()


if __name__ == "__main__":
    main()
