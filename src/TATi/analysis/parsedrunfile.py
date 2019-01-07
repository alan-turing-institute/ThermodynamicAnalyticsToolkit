#
#    ThermodynamicAnalyticsToolkit - analyze loss manifolds of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#    The TATi authors, see file AUTHORS, have asserted their moral rights.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
### 

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
