import numpy as np
import tensorflow as tf

from DataDrivenSampler.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_run(TrajectoryJob):
    ''' This implements a job that runs a new leg of a given trajectory.

    '''

    def __init__(self, data_id, continue_flag = True):
        """ Initializes a run job.

        :param data_id: id associated with data object
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_run, self).__init__(data_id)
        self.job_type = "run"
        self.continue_flag = continue_flag

    def run(self, _data, _network_model):
        """ This implements running a new leg of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :return: updated data object
        """
        # set parameters to ones from old leg (if exists)
        if len(_data.parameters) != 0:
            sess = _network_model.sess
            weights_dof = _network_model.weights.get_total_dof()
            _network_model.weights.assign(sess, _data.parameters[-1][0:weights_dof])
            _network_model.biases.assign(sess, _data.parameters[-1][weights_dof:])

        # run graph here
        run_info, trajectory = _network_model.sample(
            return_run_info=True, return_trajectories=True)

        print("Step: "+str(np.asarray(trajectory)[-1,0]))
        print("Losses: "+str(np.asarray(run_info.loc[:,'loss'])))
        print("Gradients: "+str(np.asarray(run_info.loc[:,'scaled_gradient'])))
        print("Trajectory: "+str(np.asarray(trajectory)[:,2:]))

        # append parameters to data
        _data.add_run_step(np.asarray(trajectory)[:,0],
                           np.asarray(trajectory)[:,2:],
                           np.asarray(run_info.loc[:,'loss']),
                           np.asarray(run_info.loc[:,'scaled_gradient']))

        return _data, self.continue_flag