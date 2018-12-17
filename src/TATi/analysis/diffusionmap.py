import logging
import numpy as np
import scipy.sparse as sps
import scipy

from TATi.common import setup_csv_file
import TATi.diffusion_maps.diffusionmap as dm

try:
    import pydiffmap.diffusion_map as pydiffmap_dm
    can_use_pydiffmap = True
except ImportError:
    can_use_pydiffmap = False

class DiffusionMap(object):
    """ This class contains eigenvectors and values resulting from
    the diffusion map analysis.

    """
    def __init__(self, trajectory, loss):
        self.trajectory = trajectory
        self.loss = loss

    @classmethod
    def from_parsedtrajectory(cls, parsedtrajectory):
        return cls(trajectory=parsedtrajectory.get_trajectory(),
                   loss=parsedtrajectory.get_loss())

    def compute(self, number_eigenvalues, inverse_temperature, diffusion_map_method,
                use_reweighting):
        print("Computing diffusion map")
        # NOTE: As the very first eigenvector of the diffusion map kernel is
        # constant, it is omitted in the following. To convey this to the user,
        # we start indexing at 1, not at 0, making clear that "ev_0" has been
        # omitted.
        # The first eigenvector and its eigenvalue are directly discarded by
        # :method:`compute_diffusion_maps()`, hence we only adjust the column's
        # header and the file names accordingly.
        status = True
        try:
            self.vectors, self.values, self.q = self._compute_diffusion_maps( \
                traj=self.trajectory, \
                beta=inverse_temperature, \
                loss=self.loss, \
                nrOfFirstEigenVectors=number_eigenvalues, \
                method=diffusion_map_method,
                use_reweighting=use_reweighting)
        except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence:
            print("ERROR: Vectors were non-convergent.")
            status = False
            self.vectors = np.zeros((np.shape(self.trajectory)[0], number_eigenvalues))
            self.values = np.zeros((number_eigenvalues))
            self.q = np.zeros((np.shape(self.trajectory)[0], np.shape(self.trajectory)[0]))
            # override landmarks to skip computation
        self.kernel_diff = np.asarray(self.q)
        return status

    @staticmethod
    def _get_landmarks_over_vectors(data, K, dmap_kernel, dmap_vectors, energies):
        landmark_per_vector = []
        for vindex in range(np.shape(dmap_vectors)[1]):
            V = dmap_vectors[:, vindex]

            landmarks = dm.get_landmarks(data, K, dmap_kernel, V, energies)

            landmark_per_vector.append(landmarks)
        return landmark_per_vector

    def compute_landmarks(self, num_landmarks):
        print("Getting landmarks")
        return self._get_landmarks_over_vectors( \
            data=self.trajectory, \
            K=num_landmarks, \
            dmap_kernel=self.q, \
            dmap_vectors=self.vectors, \
            energies=self.loss)

    def write_values_to_csv(self, diffusion_map_file, output_width, output_precision):
        if np.shape(self.values)[0] == 0:
            return
        if diffusion_map_file is not None:
            header = ["i", "eigenvalue"]
            csv_writer, csv_file = setup_csv_file(diffusion_map_file, header)
            for i in range(0, np.shape(self.values)[0]):
                csv_writer.writerow([i]
                                    + ['{:{width}.{precision}e}'.format(self.values[i],
                                                                        width=output_width,
                                                                        precision=output_precision)])
            csv_file.close()

    def write_vectors_to_csv(self, diffusion_matrix_file, output_width, output_precision):
        header = ["i", "loss", "kernel_diff"]
        for i in range(np.shape(self.trajectory)[1]):
            header.append("dof_" + str(i))
        for i in range(np.shape(self.vectors)[1]):
            header.append("ev_" + str(i + 1))  # we omit ev_0 as it's constant
        csv_writer, csv_file = setup_csv_file(diffusion_matrix_file, header)
        for i in range(np.shape(self.vectors)[0]):
            csv_writer.writerow([i] \
                                + ['{:{width}.{precision}e}'.format(self.loss[i, 0], \
                                                                    width=output_width, precision=output_precision)] \
                                + ['{:{width}.{precision}e}'.format(self.kernel_diff[i, 0], \
                                                                    width=output_width, precision=output_precision)] \
                                + ['{:{width}.{precision}e}'.format(x, \
                                                                    width=output_width, precision=output_precision) \
                                   for x in self.trajectory[i, :]] \
                                + ['{:{width}.{precision}e}'.format(x, \
                                                                    width=output_width, precision=output_precision) \
                                   for x in np.real(self.vectors[i, :])])
        csv_file.close()

    @staticmethod
    def _compute_diffusion_maps(traj, beta, loss, nrOfFirstEigenVectors,
                               method='vanilla', use_reweighting=False):
        if method == 'pydiffmap':
            if can_use_pydiffmap:
                if use_reweighting:
                    # pydiffmap calculates one more and leaves out the first
                    qTargetDistribution = dm.compute_target_distribution(len(traj), beta, loss)
                    # optimal choice for epsilon
                    mydmap = pydiffmap_dm.DiffusionMap(alpha=1, n_evecs=nrOfFirstEigenVectors, epsilon='bgh', k=400)
                    mydmap.fit_transform(traj, weights=qTargetDistribution)
                else:
                    mydmap = pydiffmap_dm.DiffusionMap(n_evecs=nrOfFirstEigenVectors, epsilon='bgh', k=400)
                    mydmap.fit_transform(traj)
                kernel = mydmap.kernel_matrix
                qEstimated = kernel.sum(axis=1)
                X_se = mydmap.evecs
                lambdas = mydmap.evals
            else:
                raise ValueError("Cannot use " + method + " as package not found on import.")

        elif method == 'vanilla' or method == 'TMDMap':
            epsilon = 0.1  # try 1 (i.e. make it bigger, then reduce to average distance)
            if method == "vanilla" and not use_reweighting:
                kernel = dm.compute_kernel(traj, epsilon=epsilon)
                qEstimated = kernel.sum(axis=1)
                P = dm.compute_VanillaDiffusionMap(kernel, traj)
            elif method == 'TMDMap' or (method == 'vanilla' and use_reweighting):
                qTargetDistribution = dm.compute_target_distribution(len(traj), beta, loss)
                P, qEstimated = dm.compute_TMDMap(traj, epsilon, qTargetDistribution)

            lambdas, eigenvectors = sps.linalg.eigs(P, k=nrOfFirstEigenVectors+1)  # , which='LM' )

            ix = lambdas.argsort()[::-1]
            X_se = np.real(eigenvectors[:, ix[1:]])
            lambdas = np.real(lambdas[ix[1:]])
        else:
            raise ValueError("Unknown diffusion map method "+method)

        # flip signs of ev to maximize non-negative entries (does not change ev property)
        for index in range(len(lambdas)):
            neg_signs = (X_se[:, index] < 0).sum()
            if neg_signs > X_se[:, index].size / 2:
                logging.debug("Negative signs: " + str(neg_signs) + ", dim: " + str(X_se[:, index].size)+", flipping.")
                X_se[:, index] = np.negative(X_se[:, index])
            if neg_signs == X_se[:, index].size / 2:
                # exactly half is negative, half is positive, then decide on first comp
                if X_se[0, index] < 0:
                    X_se[:, index] = np.negative(X_se[:, index])

        return X_se, lambdas, qEstimated

