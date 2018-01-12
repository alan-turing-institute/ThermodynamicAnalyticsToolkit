import numpy as np
import scipy

from DataDrivenSampler.exploration.trajectoryjob import TrajectoryJob
from DataDrivenSampler.TrajectoryAnalyser import compute_diffusion_maps

class TrajectoryJob_analyze(TrajectoryJob):
    ''' This implements a job that analyzes the last leg of a given trajectory.

    '''

    TOLERANCE = 1e-4        # tolerance for convergence of eigenvalues

    def __init__(self, data_id, continue_flag = True):
        """ Initializes an analysis job.

        :param data_id: id associated with data object
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_analyze, self).__init__(data_id)
        self.job_type = "analyze"
        self.continue_flag = continue_flag

    def run(self, _data, _object):
        """ This implements analyzing the last leg of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :param _object: FLAGS object that contains sampling parameters
        :return: updated data object
        """
        # analyze full trajectory
        trajectory = _data.parameters
        losses = _data.losses
        print("Computing diffusion map")
        try:
            vectors, values, q = compute_diffusion_maps( \
                traj=trajectory, \
                beta=_object.inverse_temperature, \
                loss=losses, \
                nrOfFirstEigenVectors=_object.number_of_eigenvalues, \
                method=_object.diffusion_map_method,
                use_reweighting=_object.use_reweighting)
        except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence:
            print("ERROR: Vectors were non-convergent.")
            vectors = np.zeros((np.shape(trajectory)[0], _object.number_of_eigenvalues))
            values = np.zeros((_object.number_of_eigenvalues))
            q = np.zeros((np.shape(trajectory)[0], np.shape(trajectory)[0]))
        kernel_diff = np.asarray(q)

        # append vectors and values to data
        _data.diffmap_eigenvectors.append(vectors)
        _data.diffmap_eigenvalues.append(values)

        # check whether converged w.r.t to previous eigenvalues
        assert( len(values) == len(_data.diffmap_eigenvalues[-1]) )
        evs_converged = True    # evs converged?
        for i in range(len(values)):
            if abs(values[i] - _data.diffmap_eigenvalues[-1][i]) > self.TOLERANCE:
                evs_converged = False

        # TODO: prune steps in leg using diffusion map

        return _data, ((not evs_converged) and (self.continue_flag))