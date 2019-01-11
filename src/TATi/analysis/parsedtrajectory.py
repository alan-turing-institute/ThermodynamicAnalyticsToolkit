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

class ParsedTrajectory(object):
    """This class encapsulates a single or multiple trajectories
    parsed from file.

    Args:

    Returns:

    """
    def __init__(self, filename, every_nth=1):
        """

        Args:
          filename: trajectory filename to parse
          every_nth: only consider every nth step (Default value = 1)

        Returns:

        """
        print("Loading trajectory file")
        self.df_trajectory = pd.read_csv(filename, sep=',', header=0)
        self.every_nth = every_nth
        self.start = 0

    def add_drop_burnin(self, drop_burnin):
        """Allows to exclude an initial selection of steps.

        Args:
          drop_burnin: up to which value in step column to exclude

        Returns:
          True - excluded, False - drop_burnin is illegal

        """
        steps = self.df_trajectory.loc[:,['step']].values
        if (len(steps) > 1) and (drop_burnin >= steps[1]):
            if drop_burnin < steps[-1]:
                self.start = next(x[0] for x in enumerate(steps) if x[1] > drop_burnin)
            else:
                return False
        return True

    def get_steps(self):
        return self.df_trajectory.loc[self.start::self.every_nth,['step']].values

    def get_loss(self):
        return self.df_trajectory.loc[self.start::self.every_nth,['loss']].values

    def get_degrees(self):
        index = self._get_weights_start(self.df_trajectory)
        return self.df_trajectory.columns[index:]

    def get_degrees_start_index(self):
        return self._get_weights_start(self.df_trajectory)

    def get_degrees_of_freedom(self):
        index = self._get_weights_start(self.df_trajectory)
        return len(self.df_trajectory.columns) - index

    def get_number_walkers(self):
        # get the range of ids in trajectory: number of distinct trajectories
        id_min = int(self.df_trajectory.min(axis=0).loc['id'])
        id_max = int(self.df_trajectory.max(axis=0).loc['id'])
        return id_max - id_min + 1 # ids start zero-based

    @staticmethod
    def _get_weights_start(dataframe):
        column_starts = ["weight0", "w0", "bias0", "b0", "coord0", "c0"]
        weight_name = None
        for i in range(len(column_starts)):
            try:
                weight_name = column_starts[i]
                weight_start_index = dataframe.columns.get_loc(weight_name)
                break
            except KeyError:
                pass
        if weight_name is None:
            raise ValueError("Could not determine start column for weights and/or biases.")
        else:
            return weight_start_index

    def get_trajectory(self):
        index = self._get_weights_start(self.df_trajectory)
        return self.df_trajectory.iloc[self.start::self.every_nth,index:].values

    def get_trajectory_for_walker(self, walker_index=None):
        index = self._get_weights_start(self.df_trajectory)
        return self.df_trajectory[
                   self.df_trajectory['id'] == walker_index].iloc[
               self.start::self.every_nth, index:].values
