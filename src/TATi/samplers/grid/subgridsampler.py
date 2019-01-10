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

import logging
import math
import numpy as np
import pandas as pd

from TATi.analysis.parsedtrajectory import ParsedTrajectory
from TATi.samplers.grid.naivegridsampler import NaiveGridSampler

class SubgridSampler(NaiveGridSampler):
    """This class constrains the naive full grid sampling approach to a
    subspace of the space of all degrees of freedom by a given set of
    directions that span the subspace.

    Args:

    Returns:

    """
    def __init__(self, network_model, exclude_parameters, samples_weights, degrees, directions):
        super(NaiveGridSampler, self).__init__(network_model=network_model,
                                               exclude_parameters=exclude_parameters)
        self.degrees = degrees
        self.directions = directions
        self.number_directions = self.directions.shape[0]
        self.samples_weights = samples_weights

        self._normalize_directions()
        if len(self.degrees) != self.weights_vals.size+self.biases_vals.size:
           raise ValueError("Directions file contains only "+str(len(self.degrees)) \
                +", while there are "+str(self.weights_vals)+" weights and "+ \
                str(self.biases_vals.size)+" biases.")

    @classmethod
    def from_file(cls, network_model, exclude_parameters, samples_weights, directions_file):
        directions = pd.read_csv(directions_file, sep=",", header=0)
        weight_start_index = ParsedTrajectory._get_weights_start(directions)
        degrees = list(directions.columns[weight_start_index:].values)
        directions = directions.iloc[:,weight_start_index:].values
        return cls(network_model=network_model,
                   exclude_parameters=exclude_parameters,
                   degrees=degrees,
                   directions=directions,
                   samples_weights=samples_weights)

    def _normalize_directions(self):
        # normalize directions
        for i in range(self.number_directions):
            self.directions[i, :] *= 1./np.linalg.norm(self.directions[i,:])

    def get_max_steps(self):
        return math.pow(self.samples_weights+1, self.number_directions)

    def _prepare_header(self):
        header = super(NaiveGridSampler, self)._prepare_header()
        for i in range(self.number_directions):
            header.append("c" + str(i))
        return header

    def setup_start(self, trajectory_file, trajectory_stepnr, interval_weights, interval_offsets):
        if trajectory_file is not None and trajectory_stepnr is not None:
            self.assign_values_from_file(filename=trajectory_file,
                                         step_nr=trajectory_stepnr)
        else:
            self.assign_values()

        # store the starting position
        self._weights_start = self.network_model.weights[0].evaluate(self._sess)
        self._biases_start = self.network_model.biases[0].evaluate(self._sess)
        print(self._weights_start)
        print(self._biases_start)
        self.interval_weights = interval_weights
        self.interval_offsets = interval_offsets

    def goto_start(self):
        super(NaiveGridSampler, self).goto_start()
        self._coords_index_grid = np.zeros(self.number_directions, dtype=int)

        if len(self.interval_offsets) != 0:
            coords_start = np.array(self.interval_offsets)
            if coords_start.size != self.number_directions:
                raise ValueError("You need to specify as many interval starts as there are degrees," + \
                    " i.e. here "+str(self.number_directions)+" dof.")
        else:
            coords_start = np.zeros(self.number_directions)
        self._coords_linspace, self._coords_len_grid = \
            self.create_linspace_grid(self.number_directions, coords_start,
                self.interval_weights, self.exclude_parameters, "w",
                self.samples_weights)

    def _check_end(self):
        isend = True
        for i in range(self.number_directions):
            if self._coords_index_grid[i] != self._coords_len_grid[i] - 1:
                isend = False
                break
        return isend

    def _next_index(self):
        for i in range(self.number_directions):
            if self._coords_index_grid[i] != self._coords_len_grid[i] - 1:
                self._coords_index_grid[i] += 1
                for j in range(i):
                    self._coords_index_grid[j] = 0
                break

    def set_step(self):
        vals = [self._coords_linspace[i][ self._coords_index_grid[i] ]
                for i in range(self.number_directions)]

        self.weights_vals = self._weights_start.copy()
        self.biases_vals = self._biases_start.copy()
        for i in range(self.number_directions):
            # set the parameters for the direction
            temp = np.multiply(self.directions[i], vals[i])
            print(temp)
            self.weights_vals += temp[:self.weights_vals.size]
            self.biases_vals += temp[self.weights_vals.size:]

            weights_eval, biases_eval = self.assign_values(do_check=True)
        return vals

    def goto_next_step(self):
        if not self._check_end():
            self._next_index()
        super(NaiveGridSampler, self).goto_next_step()

