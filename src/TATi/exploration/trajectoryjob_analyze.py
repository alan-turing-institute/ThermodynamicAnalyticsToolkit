import logging
import numpy as np
import scipy

from TATi.exploration.trajectoryjob import TrajectoryJob
from TATi.TrajectoryAnalyser import compute_diffusion_maps

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
        try:
            vectors, values, q = compute_diffusion_maps( \
                traj=trajectory, \
                beta=self.parameters.inverse_temperature, \
                loss=losses, \
                nrOfFirstEigenVectors=self.parameters.number_of_eigenvalues, \
                method=self.parameters.diffusion_map_method,
                use_reweighting=self.parameters.use_reweighting)
        except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence:
            logging.error(": Vectors were non-convergent.")
            vectors = np.zeros((np.shape(trajectory)[0], self.parameters.number_of_eigenvalues))
            values = np.zeros((self.parameters.number_of_eigenvalues))
            q = np.zeros((np.shape(trajectory)[0], np.shape(trajectory)[0]))
        kernel_diff = np.asarray(q)

        # append vectors and values to data
        _data.diffmap_eigenvectors.append(vectors)
        _data.diffmap_eigenvalues.append(values)
        logging.debug("eigenvalues is "+str(values))

        # check whether converged w.r.t to previous eigenvalues
        if len(_data.diffmap_eigenvalues) > 1:
            assert( len(values) == len(_data.diffmap_eigenvalues[-2]) )
            evs_converged = True    # evs converged?
            for i in range(len(values)):
                if abs(values[i] - _data.diffmap_eigenvalues[-2][i]) > self.TOLERANCE:
                    evs_converged = False
        else:
            evs_converged = False
        logging.debug("Has eigendecompostion converged? "+str(evs_converged))

        return _data, ((not evs_converged) and (self.continue_flag))