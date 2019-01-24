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

from TATi.models.accumulators.accumulator import Accumulator
from TATi.models.parameters.neuralnet_parameters import neuralnet_parameters


class TrajectoryAccumulator(Accumulator):
    """This class accumulates trajectory lines."""

    def __init__(self, method, config_map,
                 max_steps, every_nth, number_walkers, directions):
        super(TrajectoryAccumulator, self).__init__(method, max_steps, every_nth, number_walkers)
        self._config_map = config_map
        self._number_walkers = number_walkers
        self._directions = directions

        self.trajectory = None

    def reset(self, return_trajectories, header):
        super(TrajectoryAccumulator, self).reset()
        self._return_trajectories = return_trajectories
        self.trajectory = []
        if self._return_trajectories:
            no_params = len(header)
            for walker_index in range(self._number_walkers):
                self.trajectory.append(pd.DataFrame(
                    np.zeros((self._total_eval_steps, no_params)),
                    columns=header))

    def _accumulate_nth_step_line(self, current_step, walker_index, values):
        trajectory_line = [walker_index, values.global_step[walker_index]] \
                          + ['{:{width}.{precision}e}'.format(values.loss[walker_index],
                                                              width=self.output_width,
                                                              precision=self.output_precision)]
        dofs = []
        if len(values.weights[walker_index]) > 0:
            dofs.append(neuralnet_parameters.flatten_list_of_arrays(values.weights[walker_index]))

        if len(values.biases[walker_index]) > 0:
            dofs.append(neuralnet_parameters.flatten_list_of_arrays(values.biases[walker_index]))

        dofs = np.concatenate(dofs)
        if self._directions is not None:
            dofs = self._directions.dot(dofs)

        trajectory_line += ['{:{width}.{precision}e}'.format(item, width=self.output_width,
                                                             precision=self.output_precision) \
                            for item in dofs[:]]

        return trajectory_line

    def accumulate_nth_step(self, current_step, walker_index, values):
        if super(TrajectoryAccumulator, self).accumulate_nth_step(current_step, walker_index):
            if self._config_map["do_write_trajectory_file"] or self._return_trajectories:
                trajectory_line = self._accumulate_nth_step_line(current_step, walker_index, values)
                if self._config_map["do_write_trajectory_file"] and self._writer is not None:
                    self._writer.writerow(trajectory_line)
                if self._return_trajectories:
                    self.trajectory[walker_index].loc[self.written_row] = trajectory_line
                self.written_row +=1
