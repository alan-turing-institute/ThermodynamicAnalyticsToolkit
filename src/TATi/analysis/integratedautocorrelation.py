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

from TATi.analysis.parsedtrajectory import ParsedTrajectory

class IntegratedAutoCorrelation(object):
    """This class computes the integrated autocorrelation time on a given
    trajectory whose covariance directions have been computed beforehand.
    
    NOTE:
        When the trajectory comes from multiple walkers, i.e. there are
        multiple distinct trajectories contained in ParsedTrajectory, then
        there are two cases:
    
        1. The trajectories are averaged over
        2. The trajectories are analysed individually

    Args:

    Returns:

    """
    def __init__(self, parsedtrajectory):
        if not isinstance(parsedtrajectory, ParsedTrajectory):
            raise ValueError("IntegratedAutoCorrelation needs a ParsedTrajectory instance as cstor arg.")
        self.parsedtrajectory = parsedtrajectory
        weight_start_index = self.parsedtrajectory._get_weights_start(self.parsedtrajectory.df_trajectory)
        self.degrees = list(self.parsedtrajectory.df_trajectory.columns[weight_start_index:].values)
        self.number_degrees = len(self.degrees)
        self.number_walkers = self.parsedtrajectory.get_number_walkers()

    def _get_columns(self, _list, _id):
        return self.parsedtrajectory.df_trajectory[
                   self.parsedtrajectory.df_trajectory['id'] == _id].loc[
               self.parsedtrajectory.start::self.parsedtrajectory.every_nth, _list]

    def compute(self, transformation=None):
        if not acor_present:
            logging.critical("Required package acor could not be imported, aborting.")
            return False

        if transformation is None:
            transformation = np.eye(self.number_degrees)

        # split trajectory by ids
        values = np.array([np.asarray(self._get_columns(self.degrees, i).values)
            for i in range(self.number_walkers)])

        values = np.expand_dims(values, axis=-2)
        #print(values.shape)
        values = np.matmul(values, transformation)
        #print(values.shape)
        values = np.squeeze(values, axis=2)
        #print(values.shape)
        values = [np.mean(values[:, :, i], axis=0) for i in range(self.number_degrees)]
        #print(values[0].shape)

        # gather all values and transpose in the end
        mlist = []
        for i in range(self.number_degrees):
            try:
                mlist.append(acor.acor(values[i]))
            except RuntimeError as err:
                print("Could not compute tau for degree %d: %s\n" % (i, str(err)))
        self.tau, self.mean, self.sigma = zip(*mlist)

    def write_tau_as_csv(self, filename):
        with open(filename, "w") as of:
            of.write("weight,tau,sigma,mean\n")
            for i in range(self.number_degrees):
                of.write("%d,%lg,%lg,%lg\n" % (i, self.tau[i], self.mean[i], self.sigma[i]))
