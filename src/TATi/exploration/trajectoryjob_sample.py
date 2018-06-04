import logging
import numpy as np

from TATi.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_sample(TrajectoryJob):
    ''' This implements a job that runs a new leg of a given trajectory.

    '''

    def __init__(self, data_id, network_model, initial_step, parameters=None, continue_flag = True):
        """ Initializes a run job.

        :param data_id: id associated with data object
        :param network_model: network model containing the computational graph and session
        :param initial_step: number of first step (for continuing a trajectory)
        :param parameters: parameters of the neural net to set. If None, keep random ones
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_sample, self).__init__(data_id)
        self.job_type = "sample"
        self.network_model = network_model
        self.initial_step = initial_step
        self.parameters = parameters
        self.continue_flag = continue_flag

    def _set_parameters(self):
        # set parameters to ones from old leg (if exists)
        sess = self.network_model.sess
        # TODO: make this into a single session run call (i.e. over all replica at once)
        for replica_index in range(self.network_model.FLAGS.parallel_replica):
            weights_dof = self.network_model.weights[replica_index].get_total_dof()
            self.network_model.weights[replica_index].assign(sess, self.parameters[replica_index][0:weights_dof])
            self.network_model.biases[replica_index].assign(sess, self.parameters[replica_index][weights_dof:])

    def _set_initial_step(self):
        # set step
        for replica_index in range(self.network_model.FLAGS.parallel_replica):
            sample_step_placeholder = self.network_model.step_placeholder[replica_index]
            feed_dict = {sample_step_placeholder: self.initial_step}
            set_step = self.network_model.sess.run(
                self.network_model.global_step_assign_t[replica_index], feed_dict=feed_dict)
            logging.debug("Set initial step for replica #"+str(replica_index) \
                          +" to " + str(set_step))

    def run(self, _data):
        """ This implements running a new leg of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :return: updated data object
        """

        if self.parameters is not None:
            logging.debug("Setting initial parameters to (first ten of first replica shown) " \
                          + str(self.parameters[0][0:10]))
            self._set_parameters()

        self._set_initial_step()

        # run graph here
        self.network_model.reset_dataset()
        run_info, trajectory, averages = self.network_model.sample(
            return_run_info=True, return_trajectories=True, return_averages=True)

        self._store_trajectory(_data, averages, run_info, trajectory)

        return _data, self.continue_flag

    def _store_trajectory(self, _data, averages, run_info, trajectory):
        for replica_index in range(self.network_model.FLAGS.parallel_replica):
            steps = [int(i) for i in np.asarray(run_info[replica_index].loc[:, 'step'])]
            losses = [float(i) for i in np.asarray(run_info[replica_index].loc[:, 'loss'])]
            gradients = [float(i) for i in np.asarray(run_info[replica_index].loc[:, 'scaled_gradient'])]
            assert ("weight" not in trajectory[replica_index].columns[2])
            assert ("weight" in trajectory[replica_index].columns[3])
            parameters = np.asarray(trajectory[replica_index])[:, 3:]
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
                               _averages_lines=averages[replica_index],
                               _run_lines=run_info[replica_index],
                               _trajectory_lines=trajectory[replica_index])
