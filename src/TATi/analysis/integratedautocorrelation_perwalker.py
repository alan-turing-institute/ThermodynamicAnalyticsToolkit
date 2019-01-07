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
import logging

try:
    import acor.acor as acor
    acor_present = True
except ImportError:
    acor_present = False

from TATi.analysis.integratedautocorrelation import IntegratedAutoCorrelation

class IntegratedAutoCorrelationPerWalker(IntegratedAutoCorrelation):
    """ This class computes the integrated autocorrelation time on a given
    trajectory whose covariance directions have been computed beforehand.

    NOTE:
        When the trajectory comes from multiple walkers, i.e. there are
        multiple distinct trajectories contained in ParsedTrajectory, then
        there are two cases:

        1. The trajectories are averaged over
        2. The trajectories are analysed individually

    """

    def compute(self, transformation=None):
        if not acor_present:
            logging.critical("Required package acor could not be imported, aborting.")
            return False

        if isinstance(transformation, tuple):
            if len(transformation) != self.number_walkers:
                raise ValueError("If walker-individual transformations are given, then we need as many as there are walkers.")

        # split trajectory by ids
        values = np.array([np.asarray(self._get_columns(self.degrees, i).values) for i in range(self.number_walkers)])

        values = np.stack(values, axis=0).astype(np.float32)
        #print(values.shape)

        if isinstance(transformation, tuple):
            transformed_values = []
            for i in range(self.number_walkers):
                print(transformation[i])
                temp = np.expand_dims(values[i], axis=-2)
                transformed_values.append( np.squeeze(np.matmul(temp, transformation[i]), axis=1) )
            values = np.stack(transformed_values, axis=0).astype(np.float32)
        #print(values.shape)

        # gather all values and transpose in the end
        mlist = []
        for j in range(self.number_walkers):
            for i in range(self.number_degrees):
                try:
                    mlist.append(acor.acor(values[j, :, i]))
                except RuntimeError as err:
                    print("Could not compute tau for degree %d, walker %d: %s\n" % (i, j, str(err)))
        self.tau, self.mean, self.sigma = zip(*mlist)

    def write_tau_as_csv(self, filename):
        with open(filename, "w") as of:
            of.write("walker,weight,tau,sigma,mean\n")
            for j in range(self.number_walkers):
                for i in range(self.number_degrees):
                    index = self.number_degrees * j +i
                    of.write("%d,%d,%lg,%lg,%lg\n" % (j, i, self.tau[index], self.mean[index], self.sigma[index]))
