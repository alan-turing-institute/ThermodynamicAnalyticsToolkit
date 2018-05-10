#!/usr/bin/env python3
#
# A command-line version of Zofia's jupyter script for analysing trajectories.
#
# (C) Frederik Heber 2017-10-04

import argparse
import logging
import numpy as np
import scipy.sparse as sps
import sys

from TATi.common import str2bool
import TATi.diffusion_maps.diffusionmap as dm

try:
    import pydiffmap.diffusion_map as pydiffmap_dm
    can_use_pydiffmap = True
except ImportError:
    can_use_pydiffmap = False



from TATi.common import setup_csv_file


def parse_parameters():
    """ Sets up the argument parser for parsing command line parameters into dictionary

    :return: dictionary with parameter names as keys, unrecognized parameters
    """
    parser = argparse.ArgumentParser()
    # please adhere to alphabetical ordering
    parser.add_argument('--average_run_file', type=str, default=None,
        help='CSV file name to output averages and variances of energies.')
    parser.add_argument('--average_trajectory_file', type=str, default=None,
        help='CSV file name to output averages and variances of all degrees of freedom.')
    parser.add_argument('--diffusion_map_method', type=str, default='vanilla',
        help='Method to use for computing the diffusion map: pydiffmap, vanilla or TMDMap')
    parser.add_argument('--diffusion_map_file', type=str, default=None,
        help='Give file name to write eigenvalues of diffusion map to')
    parser.add_argument('--diffusion_matrix_file', type=str, default=None,
        help='Give file name to write eigenvectors and loss of diffusion map to')
    parser.add_argument('--drop_burnin', type=int, default=0,
        help='How many values to drop at the beginning of the trajectory.')
    parser.add_argument('--every_nth', type=int, default=1,
        help='Evaluate only every nth trajectory point to files, e.g. 10')
    parser.add_argument('--free_energy_file', type=str, default=None,
        help='Give file name ending in "-ev_1.csv" to write free energy over bins per eigenvector to')
    parser.add_argument('--inverse_temperature', type=float, default=None,
        help='Inverse temperature at which the sampling was executed for target Boltzmann distribution')
    parser.add_argument('--landmarks', type=int, default=None,
        help='How many landmark points to computer for the trajectory (if any)')
    parser.add_argument('--landmark_file', type=str, default=None,
        help='Give file name ending in "-ev_1.csv" to write trajectory at obtained landmark points per eigenvector to')
    parser.add_argument('--number_of_eigenvalues', type=int, default=4,
        help='How many largest eigenvalues to compute')
    parser.add_argument('--run_file', type=str, default=None,
        help='CSV run file name to read run time values from.')
    parser.add_argument('--steps', type=int, default=20,
        help='How many evaluation steps for averages to take')
    parser.add_argument('--trajectory_file', type=str, default=None,
        help='CSV trajectory file name to read trajectories from and compute diffusion maps on.')
    parser.add_argument('--use_reweighting', type=str2bool, default=False,
        help='Use reweighting of the kernel matrix of diffusion maps by the target distribution.')
    parser.add_argument('--version', '-V', action="store_true",
        help='Gives version information')
    return parser.parse_known_args()


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compute_diffusion_maps(traj, beta, loss, nrOfFirstEigenVectors,
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
            logging.info("Cannot use " + method + " as package not found on import.")
            sys.exit(255)
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
        logging.info("Unknown diffusion map method "+method)
        sys.exit(255)

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


def write_values_as_csv(values, csv_filename, output_width, output_precision):
    if np.shape(values)[0] == 0:
        return
    if csv_filename is not None:
        header = ["i", "eigenvalue"]
        csv_writer, csv_file = setup_csv_file(csv_filename, header)
        for i in range(0, np.shape(values)[0]):
            csv_writer.writerow([i]
                + ['{:{width}.{precision}e}'.format(values[i],
                            width=output_width,
                            precision=output_precision)])
        csv_file.close()


def write_matrix(matrix, csv_filename=None):
    with open(csv_filename, "w") as csv_file:
        matrix.tofile(csv_file)


def pdist2(x,y):
    v=np.sqrt(((x-y)**2).sum())
    return v


def get_landmarks_over_vectors(data, K, q, vectors, energies):
    landmark_per_vector = []
    for vindex in range(np.shape(vectors)[1]):
        V = vectors[:, vindex]

        landmarks = dm.get_landmarks(data, K, q, V, energies)

        landmark_per_vector.append(landmarks)
    return landmark_per_vector


def write_landmarks(traj, landmarks, csv_filename, header):
    # landmarks are just points on the trajectory that stand out, hence, logging.info the
    # trajectory at these points
    if csv_filename is not None:
        csv_writer, csv_file = setup_csv_file(csv_filename, header)
        for i in range(0, len(landmarks)):
            csv_writer.writerow(traj[i,:])
        csv_file.close()


def compute_free_energy_using_histograms(radius,   weights=None, nrbins=100, kBT=1):


    free_energy, edges=np.histogram(radius, bins=nrbins, weights = weights, normed=True)
    #free_energy+=0.0001
    free_energy= - np.log(free_energy)

    logging.debug(edges.shape)

    return free_energy, edges[:-1]


def compute_free_energy(traj, landmarks, q, vectors):
    #compute levelsets

    freeEnergies = []
    NumLevelsets = []
    for vindex in range(np.shape(vectors)[1]):
        V1 = vectors[:,vindex]
        levelsets, dummy = dm.get_levelsets(traj, landmarks, q, V1)

        K=len(levelsets)
        freeEnergy=np.zeros(K)
        h=np.zeros(K)

        for k in range(0,K):
            h[k] = len(levelsets[k])

        for k in range(0,K):
            freeEnergy[k] = -np.log(h[k]/sum(h))

        freeEnergies.append(freeEnergy)
        NumLevelsets.append(K)

    return freeEnergies, NumLevelsets
