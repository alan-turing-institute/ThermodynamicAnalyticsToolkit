import numpy as np
import scipy.sparse as sps

from TATi.analysis.parsedtrajectory import ParsedTrajectory

class Covariance(object):
    """  This class wraps the capability to perform a covariance analysis
    for a given trajectory.

    """
    def __init__(self, trajectory):
        if isinstance(trajectory, ParsedTrajectory):
            self._trajectory = trajectory.get_trajectory()
        else:
            self._trajectory = trajectory
        self._number_degrees = self._trajectory.shape[1]
        self.covariance = None
        self.values = None
        self.vectors = None

    @staticmethod
    def _setup_covariance(trajectory):
        return np.cov(trajectory, rowvar=False)

    def compute(self, number_eigenvalues):
        # set up covariance matrix
        if self.covariance is None:
            self.covariance = self._setup_covariance(self._trajectory)
        if self.values is None or self.vectors is None:
            self.values, self.vectors = self._compute_eigendecomposition(
                number_eigenvalues=number_eigenvalues, covariance=self.covariance)

    @staticmethod
    def _compute_eigendecomposition(number_eigenvalues, covariance):
        max_rank = min(covariance.shape[0], covariance.shape[1])
        if number_eigenvalues is not None and number_eigenvalues < max_rank:
            # sparse eigenvectors
            w, v = sps.linalg.eigs(covariance, k=number_eigenvalues)
        else:
            # full eigenvectors
            w, v = np.linalg.eig(covariance)
        ix = w.argsort()[::-1]
        # covariance is a symmetric real-valued matrix, hence has real-valued
        # eigensystem
        vectors = np.real(v[:, ix])
        values = np.real(w[ix])
        return values, vectors

    def write_covariance_as_csv(self, filename):
        header = [("c%d" % (i)) for i in range(self._number_degrees)]
        if filename is not None:
            np.savetxt(filename, self.covariance, delimiter=",", header=",".join(header), comments="")

    def write_vectors_as_csv(self, filename):
        # we write the vectors as transposed to have them as column vectors
        if filename is not None:
            header = [("c%d" % (i)) for i in range(self._number_degrees)]
            np.savetxt(filename, self.vectors.T, delimiter=",", header=",".join(header), comments="")

    def write_values_as_csv(self, filename):
        if filename is not None:
            np.savetxt(filename, self.values, delimiter=",", header="value", comments="")
