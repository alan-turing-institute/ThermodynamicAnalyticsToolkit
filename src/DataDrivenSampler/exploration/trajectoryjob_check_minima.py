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
    LEARNING_RATE = 1e-1 # use larger learning rate as we are close to minimum
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
            # sort the losses
            loss_at_minima_candidates = [_data.losses[i] for i in _data.minimum_candidates]
            minima_indices = sorted(range(len(loss_at_minima_candidates)), key=lambda k: loss_at_minima_candidates[k])
            print(minima_indices)
            assert( loss_at_minima_candidates[ minima_indices[0] ] <= loss_at_minima_candidates[ minima_indices[self.MAX_CANDIDATES] ] )
            if abs(loss_at_minima_candidates[ minima_indices[0] ] - loss_at_minima_candidates[ minima_indices[self.MAX_CANDIDATES] ]) < 1e-10:
                same_loss_index = 0
                for i in range(1,len(minima_indices)):
                    if abs(loss_at_minima_candidates[ minima_indices[0] ] - loss_at_minima_candidates[ minima_indices[i] ]) < 1e-10:
                        same_loss_index = minima_indices[i]
                print(same_loss_index)
                # pick MAX_CANDIDATES randomly.
                candidates = [_data.minimum_candidates[i] for i in np.random.choice(
                    minima_indices[0:same_loss_index],
                    size=self.MAX_CANDIDATES,
                    replace=False)]
                logging.info("Too many candidates with same lowest minima, we look randomly at these: "+str(candidates))
            else:
                candidates = [_data.minimum_candidates[i] for i in minima_indices[0:self.MAX_CANDIDATES]]
                logging.info("Too many candidates, we look at ones with lowest minima: "+str(candidates))
        else:
            candidates = _data.minimum_candidates
        for i in range (len(candidates)):
            parameters = _data.parameters[ candidates[i] ]
            logging.debug("Checking local minima from parameters (first ten shown) " + str(parameters[0:10]))
            sess = self.network_model.sess
            weights_dof = self.network_model.weights.get_total_dof()
            self.network_model.weights.assign(sess, parameters[0:weights_dof])
            self.network_model.biases.assign(sess, parameters[weights_dof:])

            # set step
            train_step_placeholder = self.network_model.nn.get("step_placeholder")
            feed_dict = {train_step_placeholder: candidates[i] }
            set_step = self.network_model.sess.run(self.network_model.global_step_assign_t, feed_dict=feed_dict)
            logging.debug("Set initial step to " + str(set_step))

            # run graph here
            self.network_model.reset_dataset()
            run_info, trajectory, _ = self.network_model.train(
                return_run_info=True, return_trajectories=True, return_averages=False)

            steps=[int(i) for i in np.asarray(run_info.loc[:,'step'])]
            losses=[float(i) for i in np.asarray(run_info.loc[:,'loss'])]
            gradients=[float(i) for i in np.asarray(run_info.loc[:,'scaled_gradient'])]
            assert ("weight" not in trajectory.columns[2])
            assert ("weight" in trajectory.columns[3])
            parameters=np.asarray(trajectory)[:,3:]
            logging.debug("Steps (first and last five): " + str(steps[:5]) + "\n ... \n" + str(steps[-5:]))
            logging.debug("Losses (first and last five): " + str(losses[:5]) + "\n ... \n" + str(losses[-5:]))
            logging.debug("Gradients (first and last five): " + str(gradients[:5]) + "\n ... \n" + str(gradients[-5:]))
            logging.debug("Parameters (first and last five, first ten component shown): " + str(parameters[:5][0:10]) + "\n ... \n" + str(parameters[-5:][0:10]))

            # store away found minima
            if gradients[-1] < self.GRADIENT_THRESHOLD:
                _data.local_minima.append( parameters[-1] )
                _data.loss_at_minima.append( losses[-1] )

        FLAGS.max_steps = sampler_max_steps
        FLAGS.step_width = sampler_step_width
        self.network_model.reset_parameters(FLAGS)

        return _data, self.continue_flag