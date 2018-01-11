import numpy as np
import tensorflow as tf

from DataDrivenSampler.exploration.trajectoryjob import TrajectoryJob
from DataDrivenSampler.models.neuralnet_parameters import neuralnet_parameters
from DataDrivenSampler.models.model import model as network_model

class TrajectoryJob_run(TrajectoryJob):
    ''' This implements a job that runs a new leg of a given trajectory.

    '''

    def __init__(self, data_id, network_model, continue_flag = True):
        """ Initializes a run job.

        :param data_id: id associated with data object
        :param network_model: network model containing the computational graph and session
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_run, self).__init__(data_id)
        self.job_type = "run"
        self.network_model = network_model
        self.continue_flag = continue_flag

    def run(self, _data):
        """ This implements running a new leg of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :return: updated data object
        """
        # set parameters to ones from old leg (if exists)
        if len(_data.parameters) != 0:
            sess = self.network_model.sess
            weights_dof = self.network_model.weights.get_total_dof()
            self.network_model.weights.assign(sess, _data.parameters[-1][0:weights_dof])
            self.network_model.biases.assign(sess, _data.parameters[-1][weights_dof:])

        # run graph here
        run_info, trajectory = self.network_model.sample(
            return_run_info=True, return_trajectories=True)

        step=np.asarray(trajectory)[-1,0]
        losses=[float(i) for i in np.asarray(run_info.loc[:,'loss'])]
        gradients=[float(i) for i in np.asarray(run_info.loc[:,'scaled_gradient'])]
        trajectory=np.asarray(trajectory)[:,2:]
        print("Step: "+str(step))
        print("Losses: "+str(losses))
        print("Gradients: "+str(gradients))
        print("Trajectory: "+str(trajectory))

        # append parameters to data
        _data.add_run_step(_step=step,
                           _loss=losses,
                           _gradients=gradients,
                           _parameters=trajectory,
                           _run_lines=run_info,
                           _trajectory_lines=trajectory)

        return _data, self.continue_flag