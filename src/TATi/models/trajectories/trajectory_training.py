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
    """ Refines the Trajectory class to perform a training trajectory.

    """
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

    def get_run_header(self):
        """ Prepares the distinct header for the run file for training
        """
        return ['id', 'step', 'epoch', 'accuracy', 'loss', 'time_per_nth_step', 'scaled_gradient', 'virial']

    def get_averages_header(self):
        """ Prepares the distinct header for the averages file for sampling

        """
        header = super(TrajectoryTraining, self).get_averages_header()
        if self.state.FLAGS.optimizer == "GradientDescent":
            header += ['average_virials']
        return header

    def _get_initial_test_nodes(self):
        return ["train_step"]

    def execute(self, session, dataset_dict, return_run_info = False, return_trajectories = False, return_averages=False):
        """ Performs the actual training of the neural network `nn` given a dataset `ds` and a
        Session `session`.

        :param dataset_dict: contains input_pipeline, placeholders for x and y
        :param return_run_info: if set to true, run information will be accumulated
                inside a numpy array and returned
        :param return_trajectories: if set to true, trajectories will be accumulated
                inside a numpy array and returned (adhering to FLAGS.every_nth)
        :param return_averages: if set to true, accumulated average values will be
                returned as numpy array
        :return: either twice None or a pandas dataframe depending on whether either
                parameter has evaluated to True
        """
        self.init_files()

        test_nodes = self._get_test_nodes()
        # EQN is not yet compatible with training
        #EQN_nodes =[self.state.nn[walker_index].get("EQN_step") \
        #                   for walker_index in range(self.state.FLAGS.number_walkers)]
        all_weights, all_biases = self.state._get_all_parameters()

        self.averages.reset(return_averages=return_averages,
                            header=self.get_averages_header())
        self.averages.init_writer(self.averages_writer)
        self.run_info.reset(return_run_info=return_run_info,
                            header=self.get_run_header())
        self.run_info.init_writer(self.run_writer)
        self.trajectory.reset(return_trajectories=return_trajectories,
                              header=self._get_trajectory_header())
        self.trajectory.init_writer(self.trajectory_writer)
        self.accumulated_values.reset()

        # place in feed dict: We have to supply all placeholders (regardless of
        # which the employed optimizer actually requires) because of the evaluated
        # summary! All of the placeholder nodes are also summary nodes.
        feed_dict = {}
        for walker_index in range(self.state.FLAGS.number_walkers):
            feed_dict.update(self.state._create_default_feed_dict_with_constants(walker_index))

        # check that optimizers's parameters are actually used
        self._print_parameters(session, feed_dict)

        # prepare summaries for TensorBoard
        summary_writer = self.state._prepare_summaries(session)

        # prepare some loop variables
        logging.info("Starting to train")
        logging.info_intervals = max(1, int(self.state.FLAGS.max_steps / 100))
        self.state.last_time = time.time()
        self.state.elapsed_time = 0
        if tqdm_present and self.state.FLAGS.progress:
            step_range = tqdm(range(self.state.FLAGS.max_steps))
        else:
            step_range = range(self.state.FLAGS.max_steps)

        for current_step in step_range:
            # get next batch of data
            features, labels = dataset_dict["input_pipeline"].next_batch(session)
            # logging.debug("batch is x: "+str(features[:])+", y: "+str(labels[:]))

            # update feed_dict for this step
            feed_dict.update({
                dataset_dict["xinput"]: features,
                dataset_dict["true_labels"]: labels
            })

            # zero kinetic energy and other variables
            self.state._zero_state_variables(session, self.state.FLAGS.optimizer)

            # get the weights and biases as otherwise the loss won't match
            # tf first computes loss, then gradient, then performs variable update
            # hence after the sample step, we would have updated variables but old loss
            if current_step % self.state.FLAGS.every_nth == 0:
                self.accumulated_values.weights, self.accumulated_values.biases = \
                    self._get_parameters(session, return_trajectories, all_weights, all_biases)

            ## training is not yet compatible with performing the EQN update step
            #if self.state.FLAGS.covariance_blending != 0. and \
            #        current_step % self.state.FLAGS.covariance_after_steps == 0:
            #    session.run(EQN_nodes, feed_dict=feed_dict)

            # perform the training step
            summary, self.accumulated_values.accuracy, \
            self.accumulated_values.global_step, self.accumulated_values.loss = \
                self.state._perform_step(session, test_nodes, feed_dict)

            # get updated state variables
            self.accumulated_values.evaluate(session, self.state.FLAGS.optimizer, self.state.static_vars)

            # write summaries for tensorboard
            self.state._write_summaries(summary_writer, summary, current_step)

            # accumulate averages
            if current_step >= self.state.FLAGS.burn_in_steps:
                for walker_index in range(self.state.FLAGS.number_walkers):
                    self.averages.accumulate_each_step(current_step, walker_index, self.accumulated_values)

            if current_step % self.state.FLAGS.every_nth == 0:
                self.accumulated_values.time_elapsed_per_nth_step = self.state._get_elapsed_time_per_nth_step(current_step)

            for walker_index in range(self.state.FLAGS.number_walkers):
                self.run_info.accumulate_nth_step(current_step, walker_index, self.accumulated_values)
                self.trajectory.accumulate_nth_step(current_step, walker_index, self.accumulated_values)
                self.averages.accumulate_nth_step(current_step, walker_index, self.accumulated_values)

            #if (i % logging.info_intervals) == 0:
                #logging.debug('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                #logging.debug('Loss at step %s: %s' % (i, loss_eval))

            self.state._decide_collapse_walkers(session, current_step)

        logging.info("TRAINED down to loss %s and accuracy %s." %
                     (self.accumulated_values.loss[0], self.accumulated_values.accuracy[0]))

        # close summaries file
        if self.state.FLAGS.summaries_path is not None:
            summary_writer.close()

        self.close_files()

        # get rid of possible arrays (because of multiple walkers) in return arrays
        ret_vals = [None, None, None]
        if len(self.run_info.run_info) != 0:
            ret_vals[0] = self.run_info.run_info[0]
        if len(self.trajectory.trajectory) != 0:
            ret_vals[1] = self.trajectory.trajectory[0]
        if len(self.averages.averages) != 0:
            ret_vals[2] = self.averages.averages[0]
        return ret_vals

    def init_trajectory(self, prior, model):
        # setup training/sampling
        self.optimizer = []
        for i in range(self.state.FLAGS.number_walkers):
            with tf.variable_scope("var_walker" + str(i + 1)):
                self.optimizer.append(self.state.nn[i].add_train_method(
                    model.loss[i], optimizer_method=self.state.FLAGS.optimizer,
                    prior=prior))
