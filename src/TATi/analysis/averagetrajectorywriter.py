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

import math
import numpy as np

from TATi.common import setup_csv_file
from TATi.analysis.parsedtrajectory import ParsedTrajectory

class AverageTrajectoryWriter(object):
    """This class writes the averaged parameters to a CSV file"""

    output_width = 8
    output_precision = 8

    def __init__(self, trajectory, loss=None, inverse_temperature=None):
        """

        Args:
          trajectory: trajectory to average
          loss: None or loss values to obtain average weighted by exponential
            function of negative of inverse_temperature times loss
          inverse_temperature: inverse temperature coefficient
        """
        if isinstance(trajectory, ParsedTrajectory):
            self._trajectory = trajectory.get_trajectory()
            self.header = trajectory.df_trajectory.columns
        else:
            self._trajectory = trajectory
            self.header = None
        self.number_dof = len(self._trajectory[0,:])
        weights = None
        self.loss = 0.
        if loss is not None:
            if inverse_temperature is not None:
                number_steps = loss.shape[0]
                assert( self._trajectory.shape[0] == number_steps)
                weights = np.zeros((number_steps), dtype=self._trajectory.dtype)
                for i in range(number_steps):
                    weights[i] = math.exp(-inverse_temperature*loss[i])
            else:
                self.loss = np.average(loss, weights=weights)
        self.average_params = [np.average(self._trajectory[0:, i], weights=weights)
                               for i in range(self.number_dof)]
        self.variance_params = [np.average(
            (self._trajectory[0:, i] - self.average_params[i]) ** 2,
            weights=weights)
            for i in range(self.number_dof)]
        #print("First ten parameters are converged to the following values:")
        #print(str(self.average_params[0:10]))
        #print(str(self.variance_params[0:10]))

    def write(self, filename):
        """Write average and variance of each parameter to file.

        Args:
          filename: filename of file

        Returns:

        """
        if self.header is None:
            header = ["id", "step", "loss"]+[("c%d" % i) for i in range(self.number_dof)]
        else:
            header = self.header
        csv_writer, csv_file = setup_csv_file(filename, header)
        for index, weights in enumerate([self.average_params, self.variance_params]):
            csv_writer.writerow(
                [index, int(0), self.loss] + \
                ['{:{width}.{precision}e}'.format(weight,
                                                  width=self.output_width,
                                                  precision=self.output_precision)
                 for weight in weights]
            )
        csv_file.close()
