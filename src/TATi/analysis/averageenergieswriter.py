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

from TATi.common import setup_csv_file

class AverageEnergiesWriter(object):
    """This writes averages as CSV obtained from a parsed runfile."""

    output_width = 8
    output_precision = 8

    def __init__(self, runfile, steps):
        """Creates internal averages from a given parsed run file

        Args:
          runfile: parsed runfile instance
          steps: number of steps to evaluate averages

        Returns:

        """
        self.end_list = np.arange(1,steps+1)*int(runfile.number_steps()/steps)
        print("Evaluating at steps: "+str(self.end_list))

        kinetic_energy = runfile.get_kinetic_energy()
        self.average_kinetic = [np.average(kinetic_energy[0:end]) for end in self.end_list]
        self.variance_kinetic = [np.var(kinetic_energy[0:end]) for end in self.end_list]
        loss = runfile.get_loss()
        self.average_loss = [np.average(loss[0:end]) for end in self.end_list]
        self.variance_loss = [np.var(loss[0:end]) for end in self.end_list]
        total_energy = runfile.get_total_energy()
        self.average_total = [np.average(total_energy[0:end]) for end in self.end_list]
        self.variance_total = [np.var(total_energy[0:end]) for end in self.end_list]
        self.steps = runfile.get_steps()
        print("Average first ten running kinetic energies "+str(self.average_kinetic[0:10]))

    def write(self, filename):
        csv_writer, csv_file = setup_csv_file(filename,
                                              ['step', 'average_kinetic_energy',
                                               'variance_kinetic_energy', \
                                               'average_loss', 'variance_loss', \
                                               'average_total_energy',
                                               'variance_total_energy'])
        for step, avg_kin, var_kin, avg_loss, var_loss, avg_total, var_total in zip(
                self.end_list,
                self.average_kinetic, self.variance_kinetic,
                self.average_loss, self.variance_loss,
                self.average_total, self.variance_total):
            csv_writer.writerow(
                [self.steps[step - 1]]
                + ['{:{width}.{precision}e}'.format(avg_kin, width=self.output_width,
                                                    precision=self.output_precision)] +
                ['{:{width}.{precision}e}'.format(var_kin, width=self.output_width,
                                                  precision=self.output_precision)]
                + ['{:{width}.{precision}e}'.format(avg_loss, width=self.output_width,
                                                    precision=self.output_precision)] +
                ['{:{width}.{precision}e}'.format(var_loss, width=self.output_width,
                                                  precision=self.output_precision)]
                + ['{:{width}.{precision}e}'.format(avg_total, width=self.output_width,
                                                    precision=self.output_precision)] +
                ['{:{width}.{precision}e}'.format(var_total, width=self.output_width,
                                                  precision=self.output_precision)]
            )
        csv_file.close()
