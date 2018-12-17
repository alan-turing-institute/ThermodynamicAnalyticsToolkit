import logging

import numpy as np
import scipy

from TATi.analysis.diffusionmap import DiffusionMap
from TATi.exploration.trajectoryjob import TrajectoryJob


class TrajectoryJob_analyze(TrajectoryJob):
    ''' This implements a job that analyzes the last leg of a given trajectory.

    '''

    TOLERANCE = 1e-4        # tolerance for convergence of eigenvalues

    def __init__(self, data_id, parameters, continue_flag = True):
        """ Initializes an analysis job.

        :param data_id: id associated with data object
        :param parameters: parameter for analysis
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_analyze, self).__init__(data_id)
        self.job_type = "analyze"
        self.parameters = parameters
        self.continue_flag = continue_flag

    def run(self, _data):
        """ This implements analyzing the last leg of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :param _object: FLAGS object that contains sampling parameters
        :return: updated data object
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