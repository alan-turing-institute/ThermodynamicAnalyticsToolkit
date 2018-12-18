import numpy as np
import scipy.sparse as sps

from TATi.analysis.parsedtrajectory import ParsedTrajectory

class Covariance(object):
    """  This class wraps the capability to perform a covariance analysis
    for a given trajectory.

    """
    def __init__(self, trajectory):
        if isinstance(trajectory, ParsedTrajectory):
            self.trajectory = trajectory.get_trajectory()
        else:
            self.trajectory = trajectory
        self.number_degrees = self.trajectory.shape[1]
        self.covariance = None

    def _setup_covariance(self):
        if self.covariance is None:
            self.covariance = np.cov(self.trajectory, rowvar=False)

    def compute(self, number_eigenvalues):
        self._setup_covariance()
        # set up covariance matrix
        max_rank = min(self.covariance.shape[0], self.covariance.shape[1])
        if number_eigenvalues is not None and number_eigenvalues < max_rank:
            # sparse eigenvectors
            w, v = sps.linalg.eigs(self.covariance, k=number_eigenvalues)
        else:
            # full eigenvectors
            w, v = np.linalg.eig(self.covariance)
        ix = w.argsort()[::-1]
        # covariance is a symmetric real-valued matrix, hence has real-valued
        # eigensystem
        self.vectors = np.real(v[:, ix])
        self.values = np.real(w[ix])

    def write_covariance_as_csv(self, filename):
        self._setup_covariance()
        header = [("c%d" % (i)) for i in range(self.number_degrees)]
        if filename is not None:
            np.savetxt(filename, self.covariance, delimiter=",", header=",".join(header), comments="")

    def write_vectors_as_csv(self, filename):
        # we write the vectors as transposed to have them as column vectors
        if filename is not None:
            header = [("c%d" % (i)) for i in range(self.number_degrees)]
            np.savetxt(filename, self.vectors.T, delimiter=",", header=",".join(header), comments="")

    def write_values_as_csv(self, filename):
        if filename is not None:
            np.savetxt(filename, self.values, delimiter=",", header="value", comments="")
