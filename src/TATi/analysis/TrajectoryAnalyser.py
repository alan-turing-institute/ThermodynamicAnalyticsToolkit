#!/usr/bin/env python3
#
# A command-line version of Zofia's jupyter script for analysing trajectories.
#
# (C) Frederik Heber 2017-10-04

import argparse
import logging
import numpy as np

from TATi.options.commandlineoptions import str2bool
import TATi.diffusion_maps.diffusionmap as dm

from TATi.common import setup_csv_file



def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


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


