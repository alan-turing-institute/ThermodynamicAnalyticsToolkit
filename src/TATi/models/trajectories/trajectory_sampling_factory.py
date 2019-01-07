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

from TATi.models.trajectories.trajectory_sampling_hamiltonian import TrajectorySamplingHamiltonian
from TATi.models.trajectories.trajectory_sampling_langevin import TrajectorySamplingLangevin

class TrajectorySamplingFactory(object):
    """ Factory class for creating the respective specialized TrajectorySampling
    instance.

    This is needed to avoid circular imports.
    """
    @staticmethod
    def create(samplermethod, trajectory_state):
        """ Creates a TrajectorySampling for the dynamics matching with
        the sampler specified in `samplermethod`

        :param samplermethod: sampler method
        :param trajectory_state: unique state object of trajectory, see `TrajectoryState`
        :return: created instance
        """
        if "HamiltonianMonteCarlo" in samplermethod:
            return TrajectorySamplingHamiltonian(trajectory_state)
        else:
            return TrajectorySamplingLangevin(trajectory_state)

