#!/usr/bin/env python3
#
# A command-line version of Zofia's jupyter script for analysing trajectories.
#
# (C) Frederik Heber 2017-10-04

import argparse
import numpy as np
import pandas as pd
import sys

from common import setup_csv_file


def parse_parameters():
    """ Sets up the argument parser for parsing command line parameters into dictionary

    :return: dictionary with parameter names as keys, unrecognized parameters
    """
    parser = argparse.ArgumentParser()
    # please adhere to alphabetical ordering
    parser.add_argument('--every_nth', type=int, default=1,
        help='Evaluate only every nth trajectory point to files, e.g. 10')
    parser.add_argument('--csv_file', type=str, default=None,
        help='CSV run file name to read run time values from.')
    parser.add_argument('--steps', type=int, default=20,
        help='How many evaluation steps to take')
    parser.add_argument('--version', action="store_true",
        help='Gives version information')
    parser.add_argument('--output_file', type=str, default=None,
        help='CSV file name to output averages and variances.')
    return parser.parse_known_args()


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == '__main__':
    FLAGS, unparsed = parse_parameters()

    if FLAGS.version:
        # give version and exit
        print(sys.argv[0]+" version 0.1")
        sys.exit(0)

    print("Using parameters: "+str(FLAGS))

    # load trajectory
    trajLoaded=np.asarray(pd.read_csv(FLAGS.csv_file, sep=','))

    traj=trajLoaded[::FLAGS.every_nth,:]

    loss=traj[:,2]
    kinetic_energy=traj[:,4]
    steps=traj[:,0]
    no_steps = len(steps)

    print("%d steps." % (no_steps))
    print("%lg variance." % (traj[:,1:].var()))

    end_list = np.arange(1,FLAGS.steps+1)*int(no_steps/FLAGS.steps)-1
    print(str(end_list))
    avg = [np.average(kinetic_energy[0:end]) for end in end_list]
    actual_steps = [steps[end] for end in end_list]
    print("Average final kinetic energy "+str(avg))
    #    print("Moving average kinetic energy: "+str(avg_kinetic[::10]))

    csv_writer, csv_file = setup_csv_file(FLAGS.output_file, ['step', 'average_kinetic_energy'])
    for step, avg in zip(actual_steps, avg):
        csv_writer.writerow([step, avg])
    csv_file.close()
