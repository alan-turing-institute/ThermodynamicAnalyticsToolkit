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
import numpy as np
import os
import pandas as pd
import subprocess

from TATi.exploration.trajectoryprocess import TrajectoryProcess
from TATi.exploration.get_executables import get_install_path


class TrajectoryProcess_sample(TrajectoryProcess):
    """This implements a job that runs a new leg of a given trajectory."""

    INDEX_RUN_FILENAME = 0          # index of run_file in temp_filenames
    INDEX_TRAJECTORY_FILENAME = 1   # index of trajectory in temp_filenames
    INDEX_AVERAGES_FILENAME = 2     # index of averages in temp_filenames

    def __init__(self, data_id, network_model, FLAGS, temp_filenames, restore_model, save_model=None, continue_flag = True):
        """Initializes a run job.

        Args:
          data_id: id associated with data object
          network_model: network_model containing the neural network's whole state
          FLAGS: FLAGS structure with run parameters
          temp_filenames: temporary (unique) filenames for run_info, trajectory, and averages
          restore_model: file of the model to restore from
          save_model: file of the model to save last step to (Default value = None)
          continue_flag: flag allowing to override spawning of subsequent job (Default value = True)

        Returns:

        """
        super(TrajectoryProcess_sample, self).__init__(data_id, network_model)
        self.job_type = "sample"
        self.FLAGS = FLAGS
        self.number_biases = network_model.get_total_bias_dof()
        self.number_weights = network_model.get_total_weight_dof()
        self.temp_filenames = temp_filenames
        self.restore_model = restore_model
        self.save_model = save_model
        self.continue_flag = continue_flag

    def run(self, _data):
        """This implements running a new leg of a given trajectory stored
        in a data object.

        Args:
          _data: data object to use

        Returns:
          updated data object

        """
        # construct flags for TATiampler
        sampling_flags = self.get_options_from_flags(self.FLAGS,
                   ["every_nth", "inter_ops_threads", "intra_ops_threads", "batch_data_files", \
                    "batch_data_file_type", "input_dimension", "output_dimension", \
                    "batch_size", "dropout", "fix_parameters", "hidden_activation", \
                    "hidden_dimension", "in_memory_pipeline", "input_columns", "loss", \
                    "output_activation", "prior_factor", "prior_lower_boundary", \
                    "prior_power", "prior_upper_boundary", "friction_constant", "inverse_temperature", \
                    "hamiltonian_dynamics_time", "max_steps", "sampler", "step_width"])
        # use a deterministic but different seed for each trajectory
        if self.FLAGS.seed is not None:
            sampling_flags.extend(
                ["--seed", str(self.FLAGS.seed+_data.get_id())])

        # restore initial point and save end point for next leg
        do_remove_parameter_file = False
        if self.restore_model is not None:
            foldername = os.path.dirname(self.restore_model)
            if not os.path.isdir(foldername):
                filename, step = self.create_starting_parameters(_data, self.number_weights, self.number_biases)
                sampling_flags.extend(
                    ["--parse_parameters_file", filename,
                     "--parse_steps", str(step)])
                do_remove_parameter_file = True
            else:
                sampling_flags.extend(
                    ["--restore_model", self.restore_model])
        if self.restore_model is not None:
            sampling_flags.extend(
                ["--save_model", self.save_model])
        # store all run info, trajectory and averages
        sampling_flags.extend([
            "--run_file", self.temp_filenames[self.INDEX_RUN_FILENAME],
            "--trajectory_file", self.temp_filenames[self.INDEX_TRAJECTORY_FILENAME],
            "--averages_file", self.temp_filenames[self.INDEX_AVERAGES_FILENAME]
        ])

        # run TATiampler
        p = subprocess.Popen([get_install_path()+"/TATiSampler"]+sampling_flags)
        p.communicate(None)
        retcode = p.poll()
        assert( retcode == 0 )

        # remove possibly created parameters file
        if do_remove_parameter_file:
            os.remove(filename)

        # parse files for store in _data
        run_info = pd.read_csv(self.temp_filenames[self.INDEX_RUN_FILENAME], sep=',', header=0)
        trajectory = pd.read_csv(self.temp_filenames[self.INDEX_TRAJECTORY_FILENAME], sep=',', header=0)
        averages = pd.read_csv(self.temp_filenames[self.INDEX_AVERAGES_FILENAME], sep=',', header=0)
        self._store_trajectory(_data, averages, run_info, trajectory)

        # remove run files
        for filename in self.temp_filenames:
            os.remove(filename)

        return _data, self.continue_flag

    def _store_trajectory(self, _data, averages, run_info, trajectory):
        steps = [int(i) for i in np.asarray(run_info.loc[:, 'step'])]
        losses = [float(i) for i in np.asarray(run_info.loc[:, 'loss'])]
        gradients = [float(i) for i in np.asarray(run_info.loc[:, 'scaled_gradient'])]
        assert ("weight" not in trajectory.columns[2])
        assert ("weight" in trajectory.columns[3])
        parameters = np.asarray(trajectory)[:, 3:]
        logging.debug("Steps (first and last five): " + str(steps[:5]) + "\n ... \n" + str(steps[-5:]))
        logging.debug("Losses (first and last five): " + str(losses[:5]) + "\n ... \n" + str(losses[-5:]))
        logging.debug("Gradients (first and last five): " + str(gradients[:5]) + "\n ... \n" + str(gradients[-5:]))
        logging.debug("Parameters (first and last five, first ten component shown): " + str(
            parameters[:5, 0:10]) + "\n ... \n" + str(parameters[-5:, 0:10]))
        # append parameters to data
        _data.add_run_step(_steps=steps,
                           _losses=losses,
                           _gradients=gradients,
                           _parameters=parameters,
                           _averages_lines=averages,
                           _run_lines=run_info,
                           _trajectory_lines=trajectory)
