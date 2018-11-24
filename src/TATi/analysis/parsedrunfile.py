import numpy as np
import pandas as pd

import sys

class ParsedRunfile(object):
    """ This class encapsulates a single or multiple trajectories
    parsed from file.

    """

    columns = ['step', 'loss', 'kinetic_energy', 'total_energy']

    def __init__(self, filename, every_nth):
        """

        :param filename: runfile filename to parse
        :param every_nth: only use every nth step
        """
        df_run = pd.read_csv(filename, sep=',', header=0)
        self.run = np.asarray(df_run.loc[:, self.columns])
        self.start = 0
        self.every_nth = every_nth

    def add_drop_burnin(self, drop_burnin):
        """ Allows to exclude an initial selection of steps.

        :param drop_burnin: up to which value in step column to exclude
        :return:  True - excluded, False - drop_burnin is illegal
        """
        if (len(self.run[:, 0]) > 1) and (drop_burnin >= self.run[1, 0]):
            if drop_burnin < self.run[-1, 0]:
                self.start = next(x[0] for x in enumerate(self.run[:, 0]) if x[1] > drop_burnin)
            else:
                return False
        return True
        print("Starting run array at " + str(self.start))

    def get_steps(self):
        return self.run[self.start::self.every_nth, 0]

    def get_loss(self):
        return self.run[self.start::self.every_nth, 1]

    def get_kinetic_energy(self):
        return self.run[self.start::self.every_nth, 2]

    def get_total_energy(self):
        return self.run[self.start::self.every_nth, 3]

    def number_steps(self):
        return len(self.get_steps())
