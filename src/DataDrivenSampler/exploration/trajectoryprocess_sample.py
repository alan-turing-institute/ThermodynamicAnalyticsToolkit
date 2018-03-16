import logging
import numpy as np
import os
import pandas as pd
import subprocess

from DataDrivenSampler.exploration.trajectoryprocess import TrajectoryProcess
from DataDrivenSampler.exploration.get_executables import get_install_path


class TrajectoryProcess_sample(TrajectoryProcess):
    ''' This implements a job that runs a new leg of a given trajectory.

    '''

    INDEX_RUN_FILENAME = 0          # index of run_file in temp_filenames
    INDEX_TRAJECTORY_FILENAME = 1   # index of trajectory in temp_filenames
    INDEX_AVERAGES_FILENAME = 2     # index of averages in temp_filenames

    def __init__(self, data_id, FLAGS, network_model, temp_filenames, restore_model, save_model=None, continue_flag = True):
        """ Initializes a run job.

        :param data_id: id associated with data object
        :param run_flags: FLAGS structure with run parameters
        :param temp_filenames: temporary (unique) filenames for run_info, trajectory, and averages
        :param restore_model: file of the model to restore from
        :param save_model: file of the model to save last step to
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryProcess_sample, self).__init__(data_id, network_model)
        self.job_type = "sample"
        self.FLAGS = FLAGS
        self.temp_filenames = temp_filenames
        self.restore_model = restore_model
        self.save_model = save_model
        self.continue_flag = continue_flag

    def run(self, _data):
        """ This implements running a new leg of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :return: updated data object
        """
        # construct flags for DDSampler
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
        if self.restore_model is not None:
            sampling_flags.extend(
                ["--restore_model", self.restore_model])
        sampling_flags.extend(
            ["--save_model", self.save_model]
        )
        # store all run info, trajectory and averages
        sampling_flags.extend([
            "--run_file", self.temp_filenames[self.INDEX_RUN_FILENAME],
            "--trajectory_file", self.temp_filenames[self.INDEX_TRAJECTORY_FILENAME],
            "--averages_file", self.temp_filenames[self.INDEX_AVERAGES_FILENAME]
        ])

        # run DDSampler
        p = subprocess.Popen([get_install_path()+"/DDSampler"]+sampling_flags)
        p.communicate(None)
        retcode = p.poll()
        assert( retcode == 0 )

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
