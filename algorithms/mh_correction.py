from __future__ import print_function, division
from random import *
from math import *

class MHCorrection:

    def __init__(self, U, K):
        self.U = U
        self.K = K
        self.prev_potential = None
        self.total = 0
        self.accepted = 0

    def accept_ratio(self):
        return self.accepted / max(1, self.total)

    def __call__(self, origin, target, reuse_potential=True):

        dkinetic = self.K(target[1]) - self.K(origin[1])

        potential_t = self.U(target[0])
        if not reuse_potential or self.prev_potential is None:
            potential_o = self.U(origin[0])
        else:
            potential_o = self.prev_potential
        dpotential = potential_t - potential_o

        denergy = dpotential + dkinetic

        self.total += 1
        # P(target) / P(origin) = exp(-denergy)
        if denergy <= 0.0 or exp(-denergy) >= random():
            self.accepted += 1
            self.prev_potential = potential_t
            return True
        else: return False
