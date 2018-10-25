from TATi.common import get_trajectory_header
from TATi.models.accumulators.accumulator import Accumulator
from TATi.models.neuralnet_parameters import neuralnet_parameters

import numpy as np
import pandas as pd

class TrajectoryAccumulator(Accumulator):
    """ This class accumulates trajectory lines.

    """

    def __init__(self, return_trajectories, sampler, config_map, writer,
                 header, steps, every_nth, number_walkers, directions):
        super(TrajectoryAccumulator, self).__init__(every_nth)
        self.trajectory = None
        self._return_trajectories = return_trajectories
        self._sampler = sampler
        self._config_map = config_map
        self._trajectory_writer = writer
        self._number_walkers = number_walkers
        self._directions = directions
        if self._return_trajectories:
            self.trajectory = []
            no_params = len(header)
            for walker_index in range(self._number_walkers):
                self.trajectory.append(pd.DataFrame(
                    np.zeros((steps, no_params)),
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
                if self._config_map["do_write_trajectory_file"] and self._trajectory_writer is not None:
                    self._trajectory_writer.writerow(trajectory_line)
                if self._return_trajectories:
                    self.trajectory[walker_index].loc[self.written_row] = trajectory_line
                self.written_row +=1
