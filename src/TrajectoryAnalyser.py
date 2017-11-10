#!/usr/bin/env python3
#
# A command-line version of Zofia's jupyter script for analysing trajectories.
#
# (C) Frederik Heber 2017-10-04

import argparse
import numpy as np
import scipy.sparse as sps

import DataDrivenSampler.diffusion_maps.diffusionmap as dm

from DataDrivenSampler.common import setup_csv_file


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
    parser.add_argument('--diffusion_map_file', type=str, default=None,
        help='Give file name to write eigenvalues of diffusion map to')
    parser.add_argument('--diffusion_matrix_file', type=str, default=None,
        help='Give file name to write eigenvectors and loss of diffusion map to')
    parser.add_argument('--drop_burnin', type=int, default=0,
        help='How many values to drop at the beginning of the trajectory.')
    parser.add_argument('--every_nth', type=int, default=1,
        help='Evaluate only every nth trajectory point to files, e.g. 10')
    parser.add_argument('--landmarks', type=int, default=None,
        help='How many landmark points to computer for the trajectory (if any)')
    parser.add_argument('--landmark_prefix', type=str, default=None,
        help='Give prefix to file name to write trajectory at obtained landmark points per eigenvector to')
    parser.add_argument('--run_file', type=str, default=None,
        help='CSV run file name to read run time values from.')
    parser.add_argument('--steps', type=int, default=20,
        help='How many evaluation steps to take')
    parser.add_argument('--trajectory_file', type=str, default=None,
        help='CSV trajectory file name to read trajectories from and compute diffusion maps on.')
    parser.add_argument('--version', '-V', action="store_true",
        help='Gives version information')
    return parser.parse_known_args()


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def compute_diffusion_maps(traj):
    epsilon=0.1

    #landmarks, V1 = sampler.dimension_reduction(traj, epsilon, numberOfLandmarks, smpl.model, smpl.T, method=Method)
    kernelDiff = dm.compute_kernel(traj, epsilon)
    P = dm.compute_P(kernelDiff, traj)
    q = kernelDiff.sum(axis=1)

    #P, target_distribution, qEmp = compute_unweighted_P( X, epsilon, sampler)
    lambdas, V = sps.linalg.eigsh(P, k=4)#, which='LM' )
    ix = lambdas.argsort()[::-1]
    lambdas = lambdas[ix]
    X_se = V[:,ix]

    return X_se, lambdas, q


def write_values_as_csv(values, csv_filename=None):
    if np.shape(values)[0] == 0:
        return
    if csv_filename is not None:
        header = ["i", "eigenvalue"]
        csv_writer, csv_file = setup_csv_file(csv_filename, header)
        for i in range(0, np.shape(values)[0]):
            csv_writer.writerow([i, values[i]])
        csv_file.close()


def write_matrix(matrix, csv_filename=None):
    with open(csv_filename, "w") as csv_file:
        matrix.tofile(csv_file)


def pdist2(x,y):
    v=np.sqrt(((x-y)**2).sum())
    return v


def get_landmarks(data, K, q, vectors, energies):

    landmark_per_vector = []
    for vindex in range(np.shape(vectors)[1]):
        V1 = vectors[:,vindex]
        m = float(q.size)
        #q=np.array(q)

        delta = 100/m*(max(V1)-min(V1))
        deltaMax=2*delta
        levels = np.linspace(min(V1),max(V1),num=K)

        lb = 1

        landmarks=np.zeros(K)
        emptyLevelSet=0

        for k in range(K-1, -1, -1):


                levelsetLength=0

                # we want to identify idices in V1 which are delta close to the levels
                #---o----o----o----o----o---
                #  *o** *o*  *o*  *o*   o
                # if there are no indeces in the delta distance, increase the delta distance

                while levelsetLength==0:

                    levelset = np.where(np.abs(V1 - levels[k]) < delta)
                    levelset=levelset[0]
                    levelsetLength=len(levelset)

                    delta=delta*1.001

                    if delta>deltaMax:
                        levelset=range(0,len(V1))

                data_level = data[levelset,:]

                if k==K-1:

                    idx = np.argmin(energies[levelset] / m)
                    landmarks[k]= levelset[idx]

                else:

                    idx = np.argmin(energies[levelset] / m)
                    qtmp= energies[levelset] / m

                    # compute the distance to the last landmark
                    dist_to_last=np.zeros(data_level.shape[0])
                    for i in range(0,data_level.shape[0]):
                        dist_to_last[int(i)] = pdist2(data[int(landmarks[k+1]),:], data_level[int(i)])
                    dtmp=np.array(dist_to_last.reshape(qtmp.shape))

                    v=qtmp  - lb*dtmp

                    idx = np.argmax(v);

                    landmarks[k]= levelset[idx]

        landmark_per_vector.append(landmarks.astype(int))
    return landmark_per_vector


def write_landmarks(traj, landmarks, csv_filename, header):
    # landmarks are just points on the trajectory that stand out, hence, print the
    # trajectory at these points
    if csv_filename is not None:
        csv_writer, csv_file = setup_csv_file(csv_filename, header)
        for i in range(0, len(landmarks)):
            csv_writer.writerow(traj[i,:])
        csv_file.close()


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
