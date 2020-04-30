from dataclasses import dataclass, field
from typing import List

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunResults:
    epoch_results: List[List[float]] = field(default_factory=list)

    def new_run(self):
        self.epoch_results.append([])

    def add(self, error):
        self.epoch_results[-1].append(error)

    def epoch_means(self):
        return np.array(self.epoch_results).T.mean(axis=1)


@dataclass
class Logger:
    epochs: int
    num_burn: int
    updater_results: List[RunResults] = field(default_factory=list)
    updater_names: List[str] = field(default_factory=list)

    def new_updater(self, name):
        self.updater_results.append(RunResults())
        self.updater_names.append(name)

    def new_run(self):
        self.updater_results[-1].new_run()

    def log(self, run, epoch, nll, err):
        self.updater_results[-1].add(err)
        print('updater: {}, run: {}, epoch: {}, test NLL: {:.4f}, test error: {:.4f}'
              .format(self.updater_names[-1], run, epoch, nll, err))

    def plot(self):
        plt.plot(range(self.num_burn + 1, self.epochs + 1),
                 self.updater_results[-1].epoch_means()[self.num_burn:],
                 label=self.updater_names[-1])

    def show_plot(self):
        plt.xlabel('iteration')
        plt.ylabel('test error')
        plt.xlim(0, self.epochs)
        bottom, _ = plt.ylim()
        plt.ylim(max(0, bottom), 0.05)
        plt.legend()
        plt.show()
