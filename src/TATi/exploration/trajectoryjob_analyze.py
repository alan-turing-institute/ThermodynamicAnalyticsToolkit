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
import scipy

from TATi.analysis.diffusionmap import DiffusionMap
from TATi.exploration.trajectoryjob import TrajectoryJob


class TrajectoryJob_analyze(TrajectoryJob):
    """This implements a job that analyzes the last leg of a given trajectory."""

    TOLERANCE = 1e-4        # tolerance for convergence of eigenvalues

    def __init__(self, data_id, parameters, continue_flag = True):
        """Initializes an analysis job.

        Args:
          data_id: id associated with data object
          parameters: parameter for analysis
          continue_flag: flag allowing to override spawning of subsequent job (Default value = True)

        Returns:

        """
        super(TrajectoryJob_analyze, self).__init__(data_id)
        self.job_type = "analyze"
        self.parameters = parameters
        self.continue_flag = continue_flag

    def run(self, _data):
        """This implements analyzing the last leg of a given trajectory stored
        in a data object.

        Args:
          _data: data object to use

        Returns:
          updated data object

        """
        # analyze full trajectory
        trajectory = _data.parameters
        losses = _data.losses
        logging.debug("Computing diffusion map")
        dmap = DiffusionMap( \
            trajectory=trajectory, \
            loss=losses)
        evs_converged = dmap.compute( \
            number_eigenvalues=self.parameters.number_of_eigenvalues, \
            inverse_temperature=self.parameters.inverse_temperature, \
            diffusion_map_method=self.parameters.diffusion_map_method,
            use_reweighting=self.parameters.use_reweighting)

        # append vectors and values to data
        _data.diffmap_eigenvectors.append(dmap.vectors)
        _data.diffmap_eigenvalues.append(dmap.values)
        logging.debug("eigenvalues is "+str(dmap.values))

        # check whether converged w.r.t to previous eigenvalues
        if len(_data.diffmap_eigenvalues) > 1:
            assert( len(dmap.values) == len(_data.diffmap_eigenvalues[-2]) )
            for i in range(len(dmap.values)):
                if abs(dmap.values[i] - _data.diffmap_eigenvalues[-2][i]) > self.TOLERANCE:
                    evs_converged = False
        else:
            evs_converged = False
        logging.debug("Has eigendecompostion converged? "+str(evs_converged))

        return _data, ((not evs_converged) and (self.continue_flag))