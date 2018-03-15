import logging
import numpy as np
import os
import pandas as pd
import subprocess

from DataDrivenSampler.exploration.trajectoryprocess import TrajectoryProcess

class TrajectoryProcess_train(TrajectoryProcess):
    ''' This implements a job that runs a new leg of a given trajectory
    using an Optimizer.

    '''

    INDEX_RUN_FILENAME = 0          # index of run_file in temp_filenames
    INDEX_TRAJECTORY_FILENAME = 1   # index of trajectory in temp_filenames
    INDEX_AVERAGES_FILENAME = 2     # index of averages in temp_filenames

    LEARNING_RATE = 3e-2 # use larger learning rate as we are close to minimum

    def __init__(self, data_id, FLAGS, temp_filenames, restore_model, save_model=None, continue_flag = True):
        """ Initializes a run job.

        :param data_id: id associated with data object
        :param run_flags: FLAGS structure with run parameters
        :param temp_filenames: temporary (unique) filenames for run_info, trajectory, and averages
        :param restore_model: file of the model to restore from
        :param save_model: file of the model to save last step to
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryProcess_train, self).__init__(data_id)
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

        # construct flags for DDSOptimizer
        optimizing_flags = self.get_options_from_flags(self.FLAGS,
                   ["every_nth", "inter_ops_threads", "intra_ops_threads", "batch_data_files", \
                    "batch_data_file_type", "input_dimension", "output_dimension", \
                    "batch_size", "dropout", "fix_parameters", "hidden_activation", \
                    "hidden_dimension", "in_memory_pipeline", "input_columns", "loss", \
                    "output_activation", "seed", "prior_factor", "prior_lower_boundary", \
                    "prior_power", "prior_upper_boundary", "max_steps", "optimizer"])
        # restore initial point and save end point for next leg
        optimizing_flags.extend(
            ["--step_width", str(self.LEARNING_RATE)])
        if self.restore_model is not None:
            optimizing_flags.extend(
                ["--restore_model", self.restore_model])
        optimizing_flags.extend(
            ["--save_model", self.save_model])
        # store all run info, trajectory and averages
        optimizing_flags.extend([
            "--run_file", self.temp_filenames[self.INDEX_RUN_FILENAME],
            "--trajectory_file", self.temp_filenames[self.INDEX_TRAJECTORY_FILENAME],
            "--averages_file", self.temp_filenames[self.INDEX_AVERAGES_FILENAME]
        ])

        # run DDSampler
        p = subprocess.Popen(["DDSOptimizer"]+optimizing_flags)
        p.communicate(None)
        retcode = p.poll()
        assert( retcode == 0 )

        # parse files for store in _data
        run_info = pd.read_csv(self.temp_filenames[self.INDEX_RUN_FILENAME], sep=',', header=0)
        trajectory = pd.read_csv(self.temp_filenames[self.INDEX_TRAJECTORY_FILENAME], sep=',', header=0)
        averages = pd.read_csv(self.temp_filenames[self.INDEX_AVERAGES_FILENAME], sep=',', header=0)
        self._store_last_step_of_trajectory(_data, averages, run_info, trajectory)

        # remove run files
        for filename in self.temp_filenames:
            os.remove(filename)

        return _data, self.continue_flag


    def _store_last_step_of_trajectory(self, _data, averages, run_info, trajectory):
        step = [int(np.asarray(run_info.loc[:, 'step'])[-1])]
        loss = [float(np.asarray(run_info.loc[:, 'loss'])[-1])]
        gradient = [float(np.asarray(run_info.loc[:, 'scaled_gradient'])[-1])]
        assert ("weight" not in trajectory.columns[2])
        assert ("weight" in trajectory.columns[3])
        parameters = np.asarray(trajectory)[-1:, 3:]
        logging.debug("Step : " + str(step[-1]))
        logging.debug("Loss : " + str(loss[-1]))
        logging.debug("Gradient : " + str(gradient[-1]))
        logging.debug("Parameter (first ten component shown): " + str(parameters[-1,0:10]))
        # append parameters to data
        _data.add_run_step(_steps=step,
                           _losses=loss,
                           _gradients=gradient,
                           _parameters=parameters,
                           _averages_lines=averages,
                           _run_lines=run_info,
                           _trajectory_lines=trajectory)
