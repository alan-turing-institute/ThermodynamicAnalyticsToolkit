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

from TATi.samplers.grid.trajectoryresampler import TrajectoryReSampler

class TrajectorySubspaceReSampler(TrajectoryReSampler):
    """This class implements a sampler that simply re-evaluates all points
    of a given trajectory.
    
    This can be useful if the points are to be evaluated on a different loss
    function or dataset (e.g., test data set in place of training data set).

    Args:

    Returns:

    """
    def __init__(self, network_model, exclude_parameters, steps, trajectory):
        super(TrajectorySubspaceReSampler, self).__init__(
            network_model=network_model, exclude_parameters=exclude_parameters,
            steps=steps, trajectory=trajectory)
        self.degrees = None
        self.directions = None

    @classmethod
    def from_files(cls, network_model, exclude_parameters, steps,
                   trajectory_file, directions_file):
        instance = cls.from_trajectory_file(
            network_model=network_model, exclude_parameters=exclude_parameters,
            steps=steps, trajectory_file=trajectory_file)
        instance._parse_directions_file(directions_file)
        return instance

    def _parse_directions_file(self, directions_file):
        directions = pd.read_csv(directions_file, sep=",", header=0)
        weight_start_index = self.trajectory._get_weights_start(directions)
        self.degrees = list(directions.columns[weight_start_index:].values)
        self.directions = directions.iloc[:,weight_start_index:].values
        print(self.degrees)

    def _prepare_header(self):
        header = super(TrajectoryReSampler, self)._prepare_header()
        for i in range(self.directions.shape[0]):
            header.append("c" + str(i))
        return header

    def _combine_into_coords(self, weights_eval, biases_eval):
        full_coords = super(TrajectorySubspaceReSampler, self)._combine_into_coords(
            weights_eval, biases_eval)
        subspace_coords = np.matmul(self.directions, full_coords)
        return subspace_coords
