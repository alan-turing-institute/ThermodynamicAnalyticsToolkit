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

from TATi.samplers.grid.sampler import Sampler

class NaiveGridSampler(Sampler):
    """This class implements the naive full grid sampler where each coordinate
    axis is partitioned equidistantly, i.e. the number of sampling points
    scales as O(N^d) with the number of grid points per axis N and the number
    of degrees of freedom d.

    Args:

    Returns:

    """
    def __init__(self, network_model, exclude_parameters, samples_weights, samples_biases):
        super(NaiveGridSampler, self).__init__(network_model=network_model,
                                               exclude_parameters=exclude_parameters)
        self.samples_weights = samples_weights
        self.samples_biases = samples_biases

    @staticmethod
    def getNumberFromParameter(key, name):
        start = name.index(key) + 1
        return int(name[start:])

    def get_max_steps(self):
        weight_degrees = self.weights_vals.size
        bias_degrees = self.biases_vals.size

        excluded = 0
        if (len(self.exclude_parameters) > 0):
            # subtract every degree to exclude
            for val in self.exclude_parameters:
                if "b" in val and (self.getNumberFromParameter("b",val) < self.biases_vals.size):
                    bias_degrees -= 1
                    excluded += 1
            for val in self.exclude_parameters:
                if "w" in val and (self.getNumberFromParameter("w",val) < self.weights_vals.size):
                    weight_degrees -= 1
                    excluded += 1

        print("Excluded "+str(excluded)+" parameters from max_steps sampling calculation.")

        max_steps = math.pow(self.samples_weights+1, weight_degrees)
        max_steps *= math.pow(self.samples_biases+1, bias_degrees)
        return math.ceil(max_steps)

    @staticmethod
    def create_linspace_grid(coords_size, coords_start, interval, exclude_parameters, coordname, num_samples):
        linspace = []
        len_grid = []
        for i in range(coords_size):
            interval_start = interval[0]+coords_start[i]
            interval_end = interval[1]+coords_start[i]

            keyname = coordname+str(i)
            if keyname in exclude_parameters:
                interval_start = coords_start[i]
                interval_end = interval_start

            interval_length = interval_end - interval_start

            if (num_samples > 0) and (interval_length > 0.):
                linspace.append(np.arange(0,num_samples+1)*interval_length/float(
                        num_samples)+interval_start)
                len_grid.append(num_samples+1)
            else:
                linspace.append(np.arange(0,1)+(interval_start+interval_end)/2.)
                len_grid.append(1)
        assert( len(linspace) == coords_size )
        assert( len(len_grid) == coords_size )
        #print(linspace)
        #print(len_grid)

        return linspace, len_grid

    def _prepare_header(self):
        header = super(NaiveGridSampler, self)._prepare_header()
        return self._add_all_degree_header(header)

    def setup_start(self, trajectory_file, trajectory_stepnr, interval_weights, interval_biases):
        # assign parameters to starting position from file
        if trajectory_file is not None and trajectory_stepnr is not None:
            self.assign_values_from_file(filename=trajectory_file,
                                         step_nr=trajectory_stepnr)
        else:
            self.assign_values()
        self.interval_weights = interval_weights
        self.interval_biases = interval_biases

    def goto_start(self):
        super(NaiveGridSampler, self).goto_start()
        # evaluate weights and biases to obtain interval centers from parsed file
        weights_eval = self.nn_weights.evaluate(self._sess)
        biases_eval = self.nn_biases.evaluate(self._sess)

        self._weights_index_grid = np.zeros(self.weights_vals.size, dtype=int)
        self._biases_index_grid = np.zeros(self.biases_vals.size, dtype=int)

        self._weights_linspace, self._weights_len_grid = \
            self.create_linspace_grid(self.weights_vals.size, weights_eval,
                                 [float(i) for i in self.interval_weights], self.exclude_parameters, "w",
                                      self.samples_weights)
        self._biases_linspace, self._biases_len_grid = \
            self.create_linspace_grid(self.biases_vals.size, biases_eval,
                                 [float(i) for i in self.interval_biases], self.exclude_parameters, "b",
                                      self.samples_biases)

    def _check_end(self):
        isend = True
        for i in range(self.weights_vals.size):
            if self._weights_index_grid[i] != self._weights_len_grid[i] - 1:
                isend = False
                break
        for i in range(self.biases_vals.size):
            if self._biases_index_grid[i] != self._biases_len_grid[i] - 1:
                isend = False
                break
        return isend

    def _next_index(self):
        incremented = False
        for i in range(self.weights_vals.size):
            if self._weights_index_grid[i] != self._weights_len_grid[i] - 1:
                self._weights_index_grid[i] += 1
                incremented = True
                for j in range(i):
                    self._weights_index_grid[j] = 0
                break
        if not incremented:
            for i in range(self.biases_vals.size):
                if self._biases_index_grid[i] != self._biases_len_grid[i] - 1:
                    self._biases_index_grid[i] += 1
                    for j in range(i):
                        self._biases_index_grid[j] = 0
                    for j in range(self.weights_vals.size):
                        self._weights_index_grid[j] = 0
                    break

    def set_step(self):
        # set the parameters
        self.weights_vals[:] = [self._weights_linspace[i][self._weights_index_grid[i]]
                                for i in range(self.weights_vals.size)]
        self.biases_vals[:] = [self._biases_linspace[i][self._biases_index_grid[i]]
                               for i in range(self.biases_vals.size)]

        weights_eval, biases_eval = self.assign_values(do_check=True)

        return self._combine_into_coords(weights_eval, biases_eval)

    def goto_next_step(self):
        super(NaiveGridSampler, self).goto_next_step()
        if not self._check_end():
            self._next_index()

