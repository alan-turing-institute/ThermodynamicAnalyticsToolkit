#
#    ThermodynamicAnalyticsToolkit - analyze loss manifolds of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#    The TATi authors, see file AUTHORS, have asserted their moral rights.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
### 

import logging
import numpy as np

from TATi.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_prune(TrajectoryJob):
    """This implements a job that prunes the trajectory down to a reasonable size
    while making sure that the sampled distribution is maintained.
    
    This is achieved through a metropolisation criterion, namely looking at the
    kinetic energy as an expected value which we know as we enforce the temperature.

    Args:

    Returns:

    """

    TOLERANCE = 1e-4        # tolerance for convergence of eigenvalues

    def __init__(self, data_id, network_model, continue_flag = True):
        """Initializes a prune job.

        Args:
          data_id: id associated with data object
          network_model: neural_network object
          continue_flag: flag allowing to override spawning of subsequent job (Default value = True)

        Returns:

        """
        super(TrajectoryJob_prune, self).__init__(data_id)
        self.job_type = "prune"
        self.network_model = network_model
        self.continue_flag = continue_flag

    def run(self, _data):
        """This implements pruning points from the trajectory stored in a
        data object.

        Args:
          _data: data object to use

        Returns:
          updated data object

        """
        def metropolis(old_energy, new_energy):
            logging.debug("Comparing "+str(old_energy)+" with "+str(new_energy))
            p_accept = min(1.0, np.exp(-abs(old_energy-new_energy)))
            die_roll = np.random.uniform(0.,1.)
            logging.debug("\tHaving "+str(p_accept)+" as threshold, rolled "+str(die_roll))
            return p_accept > die_roll

        num_dof = self.network_model.get_total_weight_dof()+self.network_model.get_total_bias_dof()
        FLAGS = self.network_model.get_parameters()
        average_kinetic_energy = num_dof/(2*FLAGS.inverse_temperature)

        # set a seed such that prunes occur reproducibly
        np.random.seed(FLAGS.seed+_data.get_id())

        # prune last leg
        run_lines_per_leg = _data.run_lines
        trajectory_lines_per_leg = _data.trajectory_lines
        leg_nr = -1
        run_lines = run_lines_per_leg[leg_nr]
        trajectory_lines = trajectory_lines_per_leg[leg_nr]
        keep_indices = []
        drop_indices = []
        logging.debug("Pruning of "+str(len(run_lines.index))+" steps.")
        for row in range(len(run_lines.index)):
            new_energy = run_lines.loc[run_lines.index[row], ['kinetic_energy']]
            if metropolis(average_kinetic_energy, float(np.asarray(new_energy)[0])):
                keep_indices.append(row)
            else:
                drop_indices.append(row)
        run_lines_per_leg[leg_nr] = run_lines.drop(run_lines.index[ drop_indices ])
        trajectory_lines_per_leg[leg_nr] = trajectory_lines.drop(trajectory_lines.index[ drop_indices ])

        next_leg_first_index = _data.index_at_leg[leg_nr]
        keep_indices_global = [i+next_leg_first_index for i in keep_indices]

        logging.info("Keeping "+str(len(keep_indices_global))+" of " \
            +str(len(_data.parameters))+" indices in total.")
        _data.parameters[next_leg_first_index:] = [_data.parameters[i] for i in keep_indices_global]
        _data.losses[next_leg_first_index:] = [_data.losses[i] for i in keep_indices_global]
        _data.gradients[next_leg_first_index:] = [_data.gradients[i] for i in keep_indices_global]
        assert( _data.check_size_consistency() )

        _data.is_pruned = True

        # after prune there is no other job (on that trajectory)
        return _data, False