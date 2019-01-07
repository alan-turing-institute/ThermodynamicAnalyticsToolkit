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
import pandas as pd
import scipy.sparse as sps

from TATi.analysis.parsedtrajectory import ParsedTrajectory
from TATi.analysis.covariance import Covariance

class CovariancePerWalker(Covariance):
    """  This class wraps the capability to perform a covariance analysis
    for a given trajectory.

    """
    def __init__(self, parsedtrajectory):
        if not isinstance(parsedtrajectory, ParsedTrajectory):
            raise ValueError("Covariance per walker needs a ParsedTrajector for construction")
        self.parsedtrajectory = parsedtrajectory
        self._number_degrees = self.parsedtrajectory.get_degrees_of_freedom()
        self.number_walkers = self.parsedtrajectory.get_number_walkers()
        self.covariance = [None]*self.number_walkers
        self.values = [None]*self.number_walkers
        self.vectors = [None]*self.number_walkers

    def compute(self, number_eigenvalues):
        mlist = []  # temporary storage to create values and vectors list
        for walker_index in range(self.number_walkers):
            # extract walker's trajectory
            self._trajectory = self.parsedtrajectory.get_trajectory_for_walker(walker_index)
            # set up covariance matrix
            self.covariance[walker_index] = self._setup_covariance(self._trajectory)
            # and solve eigensystem for each walker individually
            mlist.append( self._compute_eigendecomposition(
                number_eigenvalues=number_eigenvalues, covariance=self.covariance[walker_index]) )
        # transpose mlist to create result
        self.values, self.vectors = zip(*mlist)

    @staticmethod
    def _get_walker_column(_number_degrees, walker_index):
        return walker_index*np.ones((_number_degrees, 1)).astype("int")

    def write_covariance_as_csv(self, filename):
        if filename is not None:
            header = [("c%d" % i) for i in range(self._number_degrees)]
            df = pd.DataFrame(data=self.covariance[0], columns=header)
            df['id'] = 0
            for walker_index in range(1,self.number_walkers):
                df_temp = pd.DataFrame(data=self.covariance[walker_index], columns=header)
                df_temp['id'] = walker_index
                df = df.append(df_temp)
            df.to_csv(filename, sep=',', index=False, columns=['id']+header)

    def write_vectors_as_csv(self, filename):
        # we write the vectors as transposed to have them as column vectors
        if filename is not None:
            header = [("c%d" % (i)) for i in range(self._number_degrees)]
            df = pd.DataFrame(data=self.vectors[0].T, columns=header)
            df['id'] = 0
            for walker_index in range(1,self.number_walkers):
                df_temp = pd.DataFrame(data=self.vectors[walker_index].T, columns=header)
                df_temp['id'] = walker_index
                df = df.append(df_temp)
            df.to_csv(filename, sep=',', index=True, index_label='index', columns=['id'] + header)

    def write_values_as_csv(self, filename):
        if filename is not None:
            header = ["value"]
            df = pd.DataFrame(data=self.values[0], columns=header)
            df['id'] = 0
            for walker_index in range(1,self.number_walkers):
                df_temp = pd.DataFrame(data=self.values[walker_index], columns=header)
                df_temp['id'] = walker_index
                df = df.append(df_temp)
            df.to_csv(filename, sep=',', index=True, index_label='index', columns=['id'] + header)
