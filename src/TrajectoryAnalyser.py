#!/usr/bin/env python3
#
# A command-line version of Zofia's jupyter script for analysing trajectories.
#
# (C) Frederik Heber 2017-10-04

import argparse
import numpy as np


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
    parser.add_argument('--output_file', type=str, default=None,
        help='CSV file name to output averages and variances.')
    parser.add_argument('--steps', type=int, default=20,
        help='How many evaluation steps to take')
    parser.add_argument('--version', '-V', action="store_true",
        help='Gives version information')
    return parser.parse_known_args()


def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
