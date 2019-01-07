#
#    ThermodynamicAnalyticsToolkit - explore high-dimensional manifold of neural networks
#    Copyright (C) 2018 The University of Edinburgh
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

from TATi.common import setup_csv_file

class AverageTrajectoryWriter(object):
    """ This class writes the averaged parameters to a CSV file

    """

    output_width = 8
    output_precision = 8

    def __init__(self, trajectory):
        self.trajectory = trajectory
        self.number_dof = len(self.trajectory[0,:])
        self.average_params = [np.average(self.trajectory[0:, i]) for i in range(self.number_dof)]
        self.variance_params = [np.var(self.trajectory[0:, i]) for i in range(self.number_dof)]
        print("First ten parameters are converged to the following values:")
        print(str(self.average_params[0:10]))
        print(str(self.variance_params[0:10]))

    def write(self, filename):
        """ Write average and variance of each parameter to file.

        :param filename: filename of file
        """
        csv_writer, csv_file = setup_csv_file(filename, ['step', 'average_parameter', 'variance_parameter'])
        for step, avg,var in zip(range(self.number_dof), self.average_params, self.variance_params):
            csv_writer.writerow(
                [step, '{:{width}.{precision}e}'.format(avg, width=self.output_width,
                                                        precision=self.output_precision)]+
                ['{:{width}.{precision}e}'.format(var, width=self.output_width,
                                                  precision=self.output_precision)]
            )
        csv_file.close()
