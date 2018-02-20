import logging
import numpy as np

from DataDrivenSampler.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_prune(TrajectoryJob):
    ''' This implements a job that prunes the trajectory down to a reasonable size
    while making sure that the sampled distribution is maintained.

    This is achieved through a metropolisation criterion, namely looking at the
    kinetic energy as an expected value which we know as we enforce the temperature.
    '''

    TOLERANCE = 1e-4        # tolerance for convergence of eigenvalues

    def __init__(self, data_id, network_model, continue_flag = True):
        """ Initializes a prune job.

        :param data_id: id associated with data object
        :param network_model: neural_network object
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_prune, self).__init__(data_id)
        self.job_type = "prune"
        self.network_model = network_model
        self.continue_flag = continue_flag

    def run(self, _data):
        """ This implements pruning points from the trajectory stored in a
        data object.

        :param _data: data object to use
        :param _object: FLAGS object that contains sampling parameters
        :return: updated data object
        """
        def metropolis(old_energy, new_energy):
            logging.debug("Comparing "+str(old_energy)+" with "+str(new_energy))
            p_accept = min(1.0, np.exp(-abs(old_energy-new_energy)))
            die_roll = np.random.uniform(0.,1.)
            logging.debug("\tHaving "+str(p_accept)+" as threshold, rolled "+str(die_roll))
            return p_accept > die_roll

        num_dof = self.network_model.number_of_parameters
        average_kinetic_energy = num_dof/(2*self.network_model.FLAGS.inverse_temperature)

        # set a seed such that prunes occur reproducibly
        np.random.seed(self.network_model.FLAGS.seed+_data.get_id())

        # for every point on the trajectory, roll a die and evaluate criterion
        run_lines_per_leg = _data.run_lines
        trajectory_lines_per_leg = _data.trajectory_lines
        keep_indices_global = []
        for leg_nr in range(len(run_lines_per_leg)):
            run_lines = run_lines_per_leg[leg_nr]
            trajectory_lines = trajectory_lines_per_leg[leg_nr]
            keep_indices = []
            drop_indices = []
            for row in range(len(run_lines.index)):
                new_energy = run_lines.loc[run_lines.index[row], ['kinetic_energy']]
                if metropolis(average_kinetic_energy, float(np.asarray(new_energy)[0])):
                    keep_indices.append(row)
                else:
                    drop_indices.append(row)
            run_lines_per_leg[leg_nr] = run_lines.drop(run_lines.index[ drop_indices ])
            trajectory_lines_per_leg[leg_nr] = trajectory_lines.drop(trajectory_lines.index[ drop_indices ])

            keep_indices_global.extend([i+_data.index_at_leg[leg_nr] for i in keep_indices])

        logging.info("Keeping "+str(len(keep_indices_global))+" of " \
            +str(len(_data.parameters))+" indices in total.")
        _data.parameters[:] = [_data.parameters[i] for i in keep_indices_global]
        _data.losses[:] = [_data.losses[i] for i in keep_indices_global]
        _data.gradients[:] = [_data.gradients[i] for i in keep_indices_global]
        assert( _data.check_size_consistency() )

        _data.is_pruned = True

        # after prune there is no other job (on that trajectory)
        return _data, False