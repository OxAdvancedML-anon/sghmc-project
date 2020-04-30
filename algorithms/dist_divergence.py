from __future__ import print_function, division
import numpy as np
from random import *
from math import *
from scipy.stats import entropy
from scipy import integrate

import matplotlib.pyplot as plt


class DistDivergence:

    # log probability density of true distribution Q
    # samples from P will be compared against it

    # if it is not normalised the resulting cross entropy
    # will be shifted by a constant
    def __init__(self, log_pdf):
        self.log_pdf = log_pdf
        self.sum = 0.0
        self.count = 0

    # if the true distribution density is not normalised
    # this will be biased by log(A), where A is the
    # integral over R^n
    def cross_entropy(self, samples):
        # this converges to the integral of -p(x)*log(q(x)),
        # where q(x) defines the true distribution
        # and p(x) defines the sampler's distribution
        return -np.average([self.log_pdf(s) for s in samples])

    # faster variant for computing the cross entropies
    # of a growing set of samples
    def incremental_cross_entropy(self, new_samples):
        self.sum += np.sum([self.log_pdf(s) for s in new_samples])
        self.count += len(new_samples)
        return -self.sum / max(self.count, 1)

    # this will likely be shifted by a constant in the limit,
    # which is caused by errors in the normalisation of q and by
    # biases of histogram estimation of entropy
    def kld(self, samples, bins=None):
        return self.cross_entropy(samples) - sample_entropy(samples, bins)

    def incremental_kld(self, samples, new_samples, bins=None):
        return self.incremental_cross_entropy(new_samples) - sample_entropy(samples, bins)



# the bias of the estimate depends on the bin count
# https://aip.scitation.org/doi/pdf/10.1063/1.4995123
# using sqrt(|samples|) bins seems to work well
# but not tested on complex distributions

# one observation is that a low MH acceptance rate
# (caused e.g. by a noise term) significantly decreases
# the estimate, probably due to the high number of
# duplicate samples (samplers using MH all converge
# to the true distribution, so they have the same
# entropy in the limit)
def sample_entropy(samples, bins=None):
    samples = np.asarray(samples)
    n = samples.shape[0]
    dim = data_dim(samples)
    if not bins:
        bins = ceil(n**(0.5/dim))
    else:
        bins = ceil(bins**(1/dim))
    pk, edges = np.histogramdd(samples, bins=bins, density=True)
    # -sum p(x)log(p(x)) + log(w), where w = area of one bin
    # (sum(pk) is 1/w, but could compute it from edges)
    return entropy(pk.flatten()) - log(np.sum(pk))

def data_dim(samples):
    s = np.asarray(samples).squeeze()
    if s.ndim == 1: return 1
    return s.shape[1]

# q is log of unnormalised pdf (potential energy)
def integrate_volume_1d(q, samples):
    lbs = np.amin(samples)
    ubs = np.amax(samples)
    s = (ubs - lbs) / 2.0
    lbs -= s
    ubs += s
    print('-'*33)
    print('Integrating pdf over {} : {}'.format(lbs, ubs))

    np.seterr(all='ignore')
    volume, err = integrate.quad(lambda x: np.exp(-q(x)), lbs, ubs)

    print('Volume = {:.7f}, Estimated error = {:.7e}'.format(volume, err))
    print('-'*33)
    return volume

# q is log of unnormalised pdf (potential energy)
def integrate_volume_2d(q, samples):
    lbs = np.amin(samples, axis=0)
    ubs = np.amax(samples, axis=0)
    s = (ubs - lbs) / 2.0
    lbs -= s
    ubs += s
    print('-'*33)
    print('Integrating pdf over {} : {}'.format(lbs, ubs))

    np.seterr(all='ignore')
    volume, err = integrate.dblquad(lambda x, y: np.exp(-q([x, y])),
        lbs[0], ubs[0], lambda x: lbs[1], lambda x: ubs[1])

    print('Volume = {:.7f}, Estimated error = {:.7e}'.format(volume, err))
    print('-'*33)
    return volume


def test(n):
    print('Testing accuracy of entropy estimator')
    print('True distribution is N(0, 1)')
    print('Using {} samples'.format(n))

    unit_normal_ent = 0.5*log(2*pi*e)
    samples = np.random.normal(0, 1, n)

    print('True entropy = {:.7f}'.format(unit_normal_ent))
    print_entropy_table(samples, unit_normal_ent)

def print_entropy_table(samples, true_ent=None, label=None):
    samples = np.asarray(samples)
    n = samples.shape[0]
    ent_text = 'Ĥ - H'
    if true_ent is None:
        ent_text = 'Ĥ'
        true_ent = 0.0
    def_ent = sample_entropy(samples)
    if label:
        print('{:=^33}'.format(label))
    print('Est. entropy = {:.7f} (bins = sqrt(n))'.format(def_ent))
    print('-'*33)
    print('{:>7} | {:>10} | {:>10}'.format('bins', ent_text, 'log(w)'))
    print('{:>7} | {:> 10.7f} |'.format(ceil(sqrt(n)), def_ent - true_ent))
    base = 1.5
    dim = data_dim(samples)
    for i in range(ceil(1/(base-1)), ceil(log(n, base))):
        bins = ceil(base**(i/dim))
        pk, edges = np.histogramdd(samples, bins=bins, density=True)
        print('{:>7} | {:> 10.7f} | {:> 10.7f}'.
            format(bins**dim, entropy(pk.flatten()) - log(np.sum(pk)) - true_ent, -log(np.sum(pk))))
    print('-'*33)

def run_entropy_tests(extra=False):
    test(100)
    test(50000)
    if extra: test(1000000)
