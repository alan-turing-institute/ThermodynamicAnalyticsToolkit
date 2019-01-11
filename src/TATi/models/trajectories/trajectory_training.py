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
import tensorflow as tf
import time

try:
    from tqdm import tqdm # allows progress bar
    tqdm_present = True
    # workaround: otherwise we get deadlock on exceptions,
    # see https://github.com/tqdm/tqdm/issues/469
    tqdm.monitor_interval = 0
except ImportError:
    tqdm_present = False

from TATi.models.trajectories.trajectory_base import TrajectoryBase

class TrajectoryTraining(TrajectoryBase):
    """Refines the Trajectory class to perform a training trajectory."""
    def __init__(self, trajectory_state):
        super(TrajectoryTraining, self).__init__(trajectory_state)
        self.optimizer = None

    def init_accumulator(self):
        self._init_accumulator(self.state.FLAGS.optimizer)

    def _print_parameters(self, session, feed_dict):
        for walker_index in range(self.state.FLAGS.number_walkers):
            logging.info("Dependent walker #"+str(walker_index))
            deltat = session.run(self.state.nn[walker_index].get_list_of_nodes(
                ["learning_rate"]), feed_dict=feed_dict)[0]
            logging.info("GD optimizer parameters, walker #%d: delta t = %lg" % (walker_index, deltat))

    def get_methodname(self):
        return self.state.FLAGS.optimizer

    def get_run_header(self):
        """Prepares the distinct header for the run file for training"""
        return ['id', 'step', 'epoch', 'accuracy', 'loss', 'time_per_nth_step', 'scaled_gradient', 'virial']

    def get_averages_header(self):
        """Prepares the distinct header for the averages file for sampling"""
        header = super(TrajectoryTraining, self).get_averages_header()
        if self.state.FLAGS.optimizer == "GradientDescent":
            header += ['average_virials']
        return header

    def _get_initial_test_nodes(self):
        return ["train_step"]

    def init_trajectory(self, prior, model):
        # setup training/sampling
        self.optimizer = []
        for i in range(self.state.FLAGS.number_walkers):
            with tf.variable_scope("var_walker" + str(i + 1)):
                self.optimizer.append(self.state.nn[i].add_train_method(
                    model.loss[i], optimizer_method=self.state.FLAGS.optimizer,
                    prior=prior))

    @staticmethod
    def filter_execute_return_values(run_info, trajectory, averages):
        ret_vals = [None, None, None]
        if len(run_info.run_info) != 0:
            ret_vals[0] = run_info.run_info[0]
        if len(trajectory.trajectory) != 0:
            ret_vals[1] = trajectory.trajectory[0]
        if len(averages.averages) != 0:
            ret_vals[2] = averages.averages[0]
        return ret_vals

    @staticmethod
    def get_trajectory_type():
        return "train"

    @staticmethod
    def print_success(accumulated_values):
        logging.info("TRAINED, down to loss %s and accuracy %s." %
                     (accumulated_values.loss[0], accumulated_values.accuracy[0]))
