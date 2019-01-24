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

from TATi.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_check_gradient(TrajectoryJob):
    """This implements a job that checks the gradients of a given trajectory."""

    GRADIENT_THRESHOLD = 1e-5 # threshold for accepting minimum

    def __init__(self, data_id, parameters, continue_flag = True):
        """Initializes an analysis job.

        Args:
          data_id: id associated with data object
          parameters: parameter for analysis
          continue_flag: flag allowing to override spawning of subsequent job (Default value = True)

        Returns:

        """
        super(TrajectoryJob_check_gradient, self).__init__(data_id)
        self.job_type = "check_gradient"
        self.parameters = parameters
        self.continue_flag = continue_flag

    def run(self, _data):
        """This implements checking the gradient of a given trajectory stored
        in a data object.

        Args:
          _data: data object to use

        Returns:
          updated data object

        """
        # check whether last gradient is still larger than threshold
        return _data, (_data.gradients[-1] > self.GRADIENT_THRESHOLD) and self.continue_flag
