#
#    ThermodynamicAnalyticsToolkit - analyze loss manifolds of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#    The TATi authors, see file AUTHORS, have asserted their moral rights.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###

import argparse
import logging
import sys
import tensorflow as tf

from TATi.analysis.analyser_modules import AnalyserModules
from TATi.options.commandlineoptions import str2bool, react_generally_to_options
from TATi.options.pythonoptions import PythonOptions

FLAGS = None

output_width=8
output_precision=8

def get_description():
    output = "The 'modules' parameter determines which analysis functionality is executed.\n"
    output += "You have the following choices:\n"
    for name in AnalyserModules.analysis_modules:
        output += "\t"+name+"\n"
    return output

def parse_parameters():
    """ Sets up the argument parser for parsing command line parameters into dictionary

    :return: dictionary with parameter names as keys, unrecognized parameters
    """
    parser = argparse.ArgumentParser(description=get_description())
    parser.add_argument('modules', type=str, default=[], nargs='+',
        help='analysis modules to execute')
    # please adhere to alphabetical ordering
    parser.add_argument('--average_run_file', type=str, default=None,
        help='CSV file name to output averages and variances of energies.')
    parser.add_argument('--average_trajectory_file', type=str, default=None,
        help='CSV file name to output averages and variances of all degrees of freedom.')
    parser.add_argument('--covariance_matrix', type=str, default=None,
        help='Give file name to write covariance matrix as CSV to')
    parser.add_argument('--covariance_eigenvalues', type=str, default=None,
        help='Give file name to write covariance\'s eigenvalues as CSV to')
    parser.add_argument('--covariance_eigenvectors', type=str, default=None,
        help='Give file name to write covariance\'s eigenvectors as CSV to')
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
    parser.add_argument('--integrated_autocorrelation_time', type=str, default=None,
        help='Give file name to write integrated autocorrelation times to')
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
    parser.add_argument('--verbose', '-v', action='count', default=1,
        help='Level of verbosity during compare')
    parser.add_argument('--version', '-V', action="store_true",
        help='Gives version information')
    return parser.parse_known_args()


def main(_):
    global FLAGS

    modules = AnalyserModules(FLAGS, output_width, output_precision)

    modules.execute(FLAGS.modules)

def internal_main():
    global FLAGS

    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    FLAGS, unparsed = parse_parameters()

    # obtain prefix from given filename
    if FLAGS.landmark_file is not None:
        landmark_suffix = "-ev_1.csv"
        if landmark_suffix in FLAGS.landmark_file:
            FLAGS.landmark_prefix = FLAGS.landmark_file[0:FLAGS.landmark_file.find(landmark_suffix)]
        else:
            FLAGS.landmark_prefix = FLAGS.landmark_file
    else:
        FLAGS.landmark_prefix = None
    if FLAGS.free_energy_file is not None:
        free_energy_suffix = "-ev_1.csv"
        if free_energy_suffix in FLAGS.free_energy_file:
            FLAGS.free_energy_prefix = FLAGS.free_energy_file[0:FLAGS.free_energy_file.find(free_energy_suffix)]
        else:
            FLAGS.free_energy_prefix = FLAGS.free_energy_file
    else:
        FLAGS.free_energy_prefix = None

    PythonOptions.set_verbosity_level(FLAGS.verbose)
    react_generally_to_options(FLAGS, unparsed)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
