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

class TrajectoryData(object):
    """This class contains all data associated with a single trajectory.
    Trajectory steps are bundled into consecutive legs. Over these legs
    we check diffusion map values for convergence and end trajectory if this
    is the case.
    
    This is:
    -# id associated with this trajectory
    -# parameters per step
    -# loss, gradients per step
    -# eigenvectors and eigenvalues of diffusion map analysis per leg.

    Args:

    Returns:

    """

    def __init__(self, _id, _type = "sample"):
        """Initialize data object with a valid id

        Args:
          _id: id of the data object
          _type:  (Default value = "sample")

        Returns:

        """
        self.id = _id

        # these are per step
        self.steps = []
        self.parameters = []
        self.losses = []
        self.gradients = []
        self.averages_lines = []
        self.run_lines = []
        self.trajectory_lines = []

        # these are per leg
        self.legs_at_step = []  # this gives the real-world trajectory step at each leg
        self.index_at_leg = []  # this gives the offset to parameters, ... at each leg
        self.diffmap_eigenvectors = []
        self.diffmap_eigenvalues = []

        # candidates for minima
        self.minimum_candidates = []
        self.hessian_eigenvalues = []

        # indicates whether this trajectory is done
        self.is_pruned = False

        # type of dynamics that created the trajectory in this data object
        self.type = _type

        # model filename used by trajectoryprocess'es
        self.model_filename = None

    def __repr__(self):
        return ("data #"+str(self.id)+": "+str(self.steps[0:3])+"..."+str(self.steps[-3:-1]))

    def get_id(self):
        """

        Args:

        Returns:
            unique id of object

        """
        return self.id

    def add_run_step(self, _steps, _parameters, _losses, _gradients,
                     _averages_lines=None, _run_lines=None, _trajectory_lines=None):
        """Appends all values from a single run (one leg) to the specific
        internal containers for later analysis

        Args:
          _steps: step per array component
          _parameters: weight and bias) parameters of neural network as flattened vector
          _losses: loss/potential energy
          _gradients: gradient norm per step
          _averages_lines: single pandas dataframe of averages line per step (Default value = None)
          _run_lines: single pandas dataframe of run info line per step (Default value = None)
          _trajectory_lines: single pandas dataframe of trajectory line per step (Default value = None)

        Returns:

        """
        self.steps.extend(_steps)
        self.index_at_leg.append( len(self.parameters) )
        self.parameters.extend(_parameters)
        self.losses.extend(_losses)
        self.gradients.extend(_gradients)
        # trajectories need to append continuously w.r.t steps
        if (len(self.legs_at_step) > 0):
            logging.debug("Last leg ended at "+str(self.legs_at_step[-1])+", next starts at "+str(_steps[0]))
            assert( (len(self.legs_at_step) == 0) or (self.legs_at_step[-1] < _steps[0]) )
        self.legs_at_step.append(_steps[-1])
        if _averages_lines is not None:
            self.averages_lines.append(_averages_lines)
        if _run_lines is not None:
            self.run_lines.append(_run_lines)
        if _trajectory_lines is not None:
            self.trajectory_lines.append(_trajectory_lines)
        assert( self.check_size_consistency() )

    def check_size_consistency(self):
        """Checks whether the sizes of all the arrays are consistent.

        Args:

        Returns:
            True - sizes match, False - something is broken

        """
        status = True
        status = status and ( len(self.parameters) == len(self.losses) )
        status = status and ( len(self.parameters) == len(self.gradients) )
        status = status and ( len(self.legs_at_step) == len(self.index_at_leg) )
        status = status and ( (len(self.legs_at_step) == len(self.averages_lines)) \
                or (len(self.averages_lines)== 0))
        status = status and ( (len(self.legs_at_step) == len(self.run_lines)) \
                or (len(self.run_lines)== 0))
        status = status and ( (len(self.legs_at_step) == len(self.trajectory_lines) ) \
                or (len(self.trajectory_lines) == 0))
        status = status and ( len(self.diffmap_eigenvalues) == len(self.diffmap_eigenvectors) )
        return status

    def add_analyze_step(self, _eigenvectors, _eigenvalues):
        """Adds diffusion map analysis values per leg to specific containers.

        Args:
          _eigenvectors: first dominant eigenvectors of diffusion map
          _eigenvalues: first dominant eigenvalues of diffusion map

        Returns:

        """
        self.diffmap_eigenvectors.append(_eigenvectors)
        self.diffmap_eigenvalues.append(_eigenvalues)
        assert( self.check_size_consistency() )
