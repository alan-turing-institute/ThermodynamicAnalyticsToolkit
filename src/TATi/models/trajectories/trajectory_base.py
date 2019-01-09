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
import time

try:
    from tqdm import tqdm # allows progress bar
    tqdm_present = True
    # workaround: otherwise we get deadlock on exceptions,
    # see https://github.com/tqdm/tqdm/issues/469
    tqdm.monitor_interval = 0
except ImportError:
    tqdm_present = False


from TATi.common import initialize_config_map, get_trajectory_header, \
    setup_csv_file, setup_run_file, setup_trajectory_file
from TATi.models.accumulators.averagesaccumulator import AveragesAccumulator
from TATi.models.accumulators.runinfoaccumulator import RuninfoAccumulator
from TATi.models.accumulators.trajectoryaccumulator import TrajectoryAccumulator
from TATi.models.accumulators.accumulated_values import AccumulatedValues

class TrajectoryBase(object):
    """ This is the common base class for all `Trajectory`'s.

    It takes care of initializing, allowing to write lines and closing the
    average, run, and trajectory files.

    """
    def __init__(self, trajectory_state):
        self.state = trajectory_state

        # internal map for storing which file to output
        self.config_map = initialize_config_map()

        # mark writer as to be created
        self.averages_writer = None
        self.run_writer = None
        self.trajectory_writer = None

        self.init_accumulator()

    def _init_accumulator(self, methodname):
        self.averages = AveragesAccumulator(methodname,
                                       self.config_map,
                                       max_steps=self.state.FLAGS.max_steps,
                                       every_nth=self.state.FLAGS.every_nth,
                                       burn_in_steps=self.state.FLAGS.burn_in_steps,
                                       number_walkers=self.state.FLAGS.number_walkers)
        self.run_info = RuninfoAccumulator(methodname,
                                      self.config_map,
                                      max_steps=self.state.FLAGS.max_steps,
                                      every_nth=self.state.FLAGS.every_nth,
                                      number_walkers=self.state.FLAGS.number_walkers)
        self.trajectory = TrajectoryAccumulator(methodname,
                                           self.config_map,
                                           max_steps=self.state.FLAGS.max_steps,
                                           every_nth=self.state.FLAGS.every_nth,
                                           number_walkers=self.state.FLAGS.number_walkers,
                                           directions=self.state.directions)
        self.accumulated_values = AccumulatedValues()

    def get_config_map(self, key):
        if key in self.config_map.keys():
            return self.config_map[key]
        else:
            return None

    def set_config_map(self, key, value):
        self.config_map[key] = value


    def get_averages_header(self):
        """ Prepares the distinct header for the averages file for sampling

        :param setup: sample, train or None
        """
        return ['id', 'step', 'epoch', 'loss']

    def _get_trajectory_header(self):
        if self.state.directions is not None:
            print(self.directions.shape)
            header = get_trajectory_header(
                self.state.directions.shape[0],
                0)
        else:
            header = get_trajectory_header(
                self.state.weights[0].get_total_dof(),
                self.state.biases[0].get_total_dof())
        return header

    def init_files(self):
        """ Initializes the output files.
        """
        header = self.get_run_header()

        try:
            if self.averages_writer is None:
                if self.state.FLAGS.averages_file is not None:
                    self.config_map["do_write_averages_file"] = True
                    self.averages_writer, self.config_map["averages_file"] = setup_csv_file(self.state.FLAGS.averages_file, self.get_averages_header())
        except AttributeError:
            pass
        try:
            if self.run_writer is None:
                self.run_writer = setup_run_file(self.state.FLAGS.run_file, header, self.config_map)
        except AttributeError:
            pass
        try:
            if self.trajectory_writer is None:
                if self.state.directions is not None:
                    number_weights = self.state.directions.shape[0]
                    number_biases = 0
                else:
                    number_weights = self.state.weights[0].get_total_dof()
                    number_biases = self.state.biases[0].get_total_dof()
                self.trajectory_writer = setup_trajectory_file(self.state.FLAGS.trajectory_file,
                                                               number_weights,
                                                               number_biases,
                                                               self.config_map)
        except AttributeError:
            pass

    def write_run_row(self, line):
        self.run_writer.writerow(line)

    def write_trajectory_row(self, line):
        self.trajectory_writer.writerow(line)

    def write_averages_row(self, line):
        self.averages_writer.writerow(line)

    def close_files(self):
        """ Closes the output files if they have been opened.
        """
        if self.config_map["do_write_averages_file"]:
            assert self.config_map["averages_file"] is not None
            self.config_map["averages_file"].close()
            self.config_map["averages_file"] = None
            self.averages_writer = None
        if self.config_map["do_write_run_file"]:
            assert self.config_map["csv_file"] is not None
            self.config_map["csv_file"].close()
            self.config_map["csv_file"] = None
            self.run_writer = None
        if self.config_map["do_write_trajectory_file"]:
            assert self.config_map["trajectory_file"] is not None
            self.config_map["trajectory_file"].close()
            self.config_map["trajectory_file"] = None
            self.trajectory_writer = None

    def _get_test_nodes(self):
        """ Helper function to create list of nodes for activating sampling or
        training step.

        :param setup: sample or train
        """
        list_of_nodes = self._get_initial_test_nodes()
        list_of_nodes.extend(["accuracy", "global_step", "loss"])
        if self.state.FLAGS.summaries_path is not None:
            test_nodes = [self.state.summary]*self.state.FLAGS.number_walkers
        else:
            test_nodes = []
        for item in list_of_nodes:
            test_nodes.append([self.state.nn[walker_index].get(item) \
                               for walker_index in range(self.state.FLAGS.number_walkers)])
        return test_nodes

    def _get_parameters(self, session, return_trajectories, all_weights, all_biases):
        weights_eval, biases_eval = None, None
        if self.config_map["do_write_trajectory_file"] or return_trajectories:
            weights_eval, biases_eval = session.run([all_weights, all_biases])
            # [logging.info(str(item)) for item in weights_eval]
            # [logging.info(str(item)) for item in biases_eval]
        return weights_eval, biases_eval

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

        placeholder_nodes = self.get_placeholder_nodes()

        test_nodes = self._get_test_nodes()
        all_weights, all_biases = self.state._get_all_parameters()
        parameters_list = [return_trajectories, all_weights, all_biases]

        self.averages.reset(return_averages=return_averages,
                            header=self.get_averages_header())
        self.averages.init_writer(self.averages_writer)
        self.set_beta_for_execute()
        self.run_info.reset(return_run_info=return_run_info,
                            header=self.get_run_header())
        self.run_info.init_writer(self.run_writer)
        self.trajectory.reset(return_trajectories=return_trajectories,
                              header=self._get_trajectory_header())
        self.trajectory.init_writer(self.trajectory_writer)
        self.accumulated_values.reset()

        # place in feed dict: We have to supply all placeholders (regardless of
        # which the employed optimizer/sampler actually requires) because of the evaluated
        # summary! All of the placeholder nodes are also summary nodes.
        feed_dict = {}
        for walker_index in range(self.state.FLAGS.number_walkers):
            feed_dict.update(self.state._create_default_feed_dict_with_constants(walker_index))

        self.zero_extra_nodes(session)

        # check that optimizer/sampler's parameters are actually used
        self._print_parameters(session, feed_dict)

        # prepare summaries for TensorBoard
        summary_writer = self.state._prepare_summaries(session)

        # prepare some loop variables
        logging.info("Starting to " + str(self.get_trajectory_type()))
        logging.info_intervals = max(1, int(self.state.FLAGS.max_steps / 100))
        self.state.last_time = time.time()
        self.state.elapsed_time = 0
        extra_values = self.prepare_extra_values(placeholder_nodes, session, feed_dict)

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
            self.update_feed_dict(feed_dict, placeholder_nodes, current_step)

            # some extra operations just before the actual update step
            self.extra_evaluation_before_step(current_step, session, placeholder_nodes, test_nodes, feed_dict, extra_values)

            # zero kinetic energy and other variables
            self.state._zero_state_variables(session, self.get_methodname())

            # get the weights and biases as otherwise the loss won't match
            # tf first computes loss, then gradient, then performs variable update
            # hence after the sample step, we would have updated variables but old loss
            if current_step % self.state.FLAGS.every_nth == 0:
                self.accumulated_values.weights, self.accumulated_values.biases = \
                    self._get_parameters(session, return_trajectories, all_weights, all_biases)

            # perform the sampling step
            summary, self.accumulated_values.accuracy, \
            self.accumulated_values.global_step, self.accumulated_values.loss =\
                self.perform_step(current_step, session, test_nodes, feed_dict)

            # get updated state variables
            self.accumulated_values.evaluate(session, self.get_methodname(), self.state.static_vars)

            # give output on debug mode
            self.print_energies(current_step, extra_values)

            # react to updated values if necessary
            self.update_values(current_step, session, test_nodes, feed_dict, extra_values, parameters_list)

            # write summaries for tensorboard
            self.state._write_summaries(summary_writer, summary, current_step)

            # accumulate averages
            self.update_averages(current_step)

            if current_step % self.state.FLAGS.every_nth == 0:
                self.accumulated_values.time_elapsed_per_nth_step = self.state._get_elapsed_time_per_nth_step(current_step)

            for walker_index in range(self.state.FLAGS.number_walkers):
                self.run_info.accumulate_nth_step(current_step, walker_index, self.accumulated_values)
                self.trajectory.accumulate_nth_step(current_step, walker_index, self.accumulated_values)
                self.averages.accumulate_nth_step(current_step, walker_index, self.accumulated_values)

            #if (i % logging.info_intervals) == 0:
                #logging.debug('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                #logging.debug('Loss at step %s: %s' % (i, loss_eval))

            # any last updates
            self.update_extra_values(extra_values)

            self.state._decide_collapse_walkers(session, current_step)

        self.print_success(self.accumulated_values)

        # close summaries file
        if self.state.FLAGS.summaries_path is not None:
            summary_writer.close()

        self.close_files()

        return self.filter_execute_return_values(self.run_info, self.trajectory, self.averages)

    def get_placeholder_nodes(self):
        pass

    def set_beta_for_execute(self):
        pass

    def perform_step(self, current_step, session, test_nodes, feed_dict):
        return self.state._perform_step(session, test_nodes, feed_dict)

    def zero_extra_nodes(self, session):
        pass

    def prepare_extra_values(self, placeholder_nodes, session, feed_dict):
        pass

    def update_feed_dict(self, feed_dict, placeholder_nodes, current_step):
        pass

    def extra_evaluation_before_step(self, current_step, session, placeholder_nodes, test_nodes, feed_dict, extra_values):
        pass

    def print_energies(self, current_step, extra_values):
        pass

    def update_averages(self, current_step):
        if current_step >= self.state.FLAGS.burn_in_steps:
            for walker_index in range(self.state.FLAGS.number_walkers):
                self.averages.accumulate_each_step(current_step, walker_index, self.accumulated_values)

    def update_extra_values(self, extra_values):
        pass

    def update_values(self, current_step, session, test_nodes, feed_dict, extra_values, parameter_list):
        pass

    @staticmethod
    def filter_execute_return_values(run_info, trajectory, averages):
        return run_info.run_info, trajectory.trajectory, averages.averages

