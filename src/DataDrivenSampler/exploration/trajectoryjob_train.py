import logging
import numpy as np
from tensorflow.python.framework import errors_impl

import tensorflow as tf

from DataDrivenSampler.exploration.trajectoryjob_sample import TrajectoryJob_sample

class TrajectoryJob_train(TrajectoryJob_sample):
    ''' This implements a job that runs a new leg of a given trajectory.

    '''

    LEARNING_RATE = 3e-2 # use larger learning rate as we are close to minimum

    def __init__(self, data_id, network_model, initial_step, parameters=None, continue_flag = True):
        """ Initializes a run job.

        :param data_id: id associated with data object
        :param network_model: network model containing the computational graph and session
        :param initial_step: number of first step (for continuing a trajectory)
        :param parameters: parameters of the neural net to set. If None, keep random ones
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_train, self).__init__(
            data_id=data_id,
            network_model=network_model,
            initial_step=initial_step,
            parameters=parameters,
            continue_flag=continue_flag)
        self.job_type = "train"

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

    def run(self, _data):
        """ This implements running a new leg of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :return: updated data object
        """
        # modify max_steps for optimization
        FLAGS = self.network_model.FLAGS
        sampler_step_width = FLAGS.step_width
        FLAGS.step_width = self.LEARNING_RATE
        self.network_model.reset_parameters(FLAGS)

        if self.parameters is not None:
            logging.debug("Checking local minima from parameters (first ten shown) " + str(self.parameters[0:10]))
            self._set_parameters()

        self._set_initial_step()

        # run graph here
        self.network_model.reset_dataset()
        try:
            run_info, trajectory, averages = self.network_model.train(
                return_run_info=True, return_trajectories=True, return_averages=True)

            self._store_last_step_of_trajectory(_data, averages, run_info, trajectory)

        except errors_impl.InvalidArgumentError:
            logging.error("The trajectory diverged, aborting.")
            self.continue_flag = False

        # compute hessian if desired
        # get next batch of data
        if self.network_model.FLAGS.do_hessians:
            self.network_model.reset_dataset()
            features, labels = self.network_model.input_pipeline.next_batch(self.network_model.sess)
            feed_dict = {
                self.network_model.xinput: features,
                self.network_model.nn.placeholder_nodes["y_"]: labels,
            }
            if self.network_model.FLAGS.dropout is not None:
                feed_dict.update({
                    self.network_model.nn.placeholder_nodes["keep_prob"]: self.FLAGS.dropout})

            #self.network_model.reset_dataset()
            # gradient_eval = self.network_model.sess.run(
            #     self.network_model.gradients,
            #     feed_dict=feed_dict)
            # print(gradient_eval)

            hessian_eval = self.network_model.sess.run(
                self.network_model.hessians,
                feed_dict=feed_dict)
            #print(hessian_eval)
            _data.hessian_eigenvalues.append(np.linalg.eigvals(hessian_eval))
            logging.info("Max and min eigenvalues of hessian: "+str(_data.hessian_eigenvalues[0:2]) \
                         +"..."+str(_data.hessian_eigenvalues[-3:-1]))

        # set FLAGS back to old values
        FLAGS.step_width = sampler_step_width
        self.network_model.reset_parameters(FLAGS)

        return _data, self.continue_flag
