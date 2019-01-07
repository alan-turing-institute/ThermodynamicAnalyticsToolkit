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

