import logging
import numpy as np
import tensorflow as tf

from DataDrivenSampler.exploration.trajectoryjob import TrajectoryJob
from DataDrivenSampler.models.neuralnet_parameters import neuralnet_parameters
from DataDrivenSampler.models.model import model as network_model

class TrajectoryJob_check_minima(TrajectoryJob):
    ''' This implements a job that runs a new leg of a given trajectory.

    '''

    EQUILIBRATION_STEPS = 1000 # number of steps for minimum search
    LEARNING_RATE = 1 # use larger learning rate as we are close to minimum
    MAX_CANDIDATES = 3 # dont check more than this number of candidates
    GRADIENT_THRESHOLD = 1e-5 # threshold for accepting minimum

    def __init__(self, data_id, network_model, continue_flag = True):
        """ Initializes a run job.

        :param data_id: id associated with data object
        :param network_model: network model containing the computational graph and session
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_check_minima, self).__init__(data_id)
        self.job_type = "check_minima"
        self.network_model = network_model
        self.continue_flag = continue_flag

    def run(self, _data):
        """ This implements running a new leg of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :return: updated data object
        """
        # modify max_steps for optimization
        FLAGS = self.network_model.FLAGS
        sampler_max_steps = FLAGS.max_steps
        sampler_step_width = FLAGS.step_width
        FLAGS.max_steps = self.EQUILIBRATION_STEPS
        FLAGS.step_width = self.LEARNING_RATE
        self.network_model.reset_parameters(FLAGS)

        # set parameters to ones from old leg (if exists)
        if len(_data.minimum_candidates) > self.MAX_CANDIDATES:
            # pick MAX_CANDIDATES randomly.
            candidates = np.random.choice(
                _data.minimum_candidates,
                size=self.MAX_CANDIDATES,
                replace=False)
            logging.info("Too many candidates, we look at these: "+str(candidates))
        else:
            candidates = _data.minimum_candidates
        for i in range (len(candidates)):
            parameters = _data.parameters[ candidates[i] ]
            sess = self.network_model.sess
            weights_dof = self.network_model.weights.get_total_dof()
            self.network_model.weights.assign(sess, parameters[0:weights_dof])
            self.network_model.biases.assign(sess, parameters[weights_dof:])

            # run graph here
            run_info, trajectory = self.network_model.train(
                return_run_info=True, return_trajectories=True)

            steps=[int(i) for i in np.asarray(run_info.loc[:,'step'])]
            losses=[float(i) for i in np.asarray(run_info.loc[:,'loss'])]
            gradients=[float(i) for i in np.asarray(run_info.loc[:,'scaled_gradient'])]
            parameters=np.asarray(trajectory)[:,2:]
            print("Steps (last ten): "+str(steps[-10:]))
            print("Losses (last ten): "+str(losses[-10:]))
            print("Gradients (last ten): "+str(gradients[-10:]))
            print("Parameters (last ten): "+str(parameters[-10:]))

            # store away found minima
            if gradients[-1] < self.GRADIENT_THRESHOLD:
                _data.local_minima.append( parameters[-1] )
                _data.loss_at_minima.append( losses[-1] )

        FLAGS.max_steps = sampler_max_steps
        FLAGS.step_width = sampler_step_width
        self.network_model.reset_parameters(FLAGS)

        return _data, self.continue_flag