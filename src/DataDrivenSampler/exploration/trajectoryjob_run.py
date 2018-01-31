import numpy as np
import tensorflow as tf

from DataDrivenSampler.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_run(TrajectoryJob):
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
        super(TrajectoryJob_run, self).__init__(data_id)
        self.job_type = "run"
        self.network_model = network_model
        self.initial_step = initial_step
        self.parameters = parameters
        self.continue_flag = continue_flag

    def run(self, _data):
        """ This implements running a new leg of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :return: updated data object
        """
        # set parameters to ones from old leg (if exists)
        if self.parameters is not None:
            sess = self.network_model.sess
            weights_dof = self.network_model.weights.get_total_dof()
            self.network_model.weights.assign(sess, self.parameters[0:weights_dof])
            self.network_model.biases.assign(sess, self.parameters[weights_dof:])

        # set step
        sample_step_placeholder = self.network_model.nn.get("step_placeholder")
        feed_dict = {sample_step_placeholder: self.initial_step}
        set_step = self.network_model.sess.run(self.network_model.global_step_assign_t, feed_dict=feed_dict)
        print("Set initial step to " + str(set_step))

        # run graph here
        run_info, trajectory = self.network_model.sample(
            return_run_info=True, return_trajectories=True)

        steps=[int(i) for i in np.asarray(run_info.loc[:,'step'])]
        losses=[float(i) for i in np.asarray(run_info.loc[:,'loss'])]
        gradients=[float(i) for i in np.asarray(run_info.loc[:,'scaled_gradient'])]
        parameters=np.asarray(trajectory)[:,2:]
        print("Steps (last ten): "+str(steps[-10:]))
        print("Losses (last ten): "+str(losses[-10:]))
        print("Gradients (last ten): "+str(gradients[-10:]))
        print("Parameters (last ten): "+str(parameters[-10:]))

        # append parameters to data
        _data.add_run_step(_steps=steps,
                           _losses=losses,
                           _gradients=gradients,
                           _parameters=parameters,
                           _run_lines=run_info,
                           _trajectory_lines=trajectory)

        return _data, self.continue_flag