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
import csv
import logging
import numpy as np
import sys

from TATi.datasets.classificationdatasets import ClassificationDatasets as DatasetGenerator
from TATi.options.commandlineoptions import react_generally_to_options

FLAGS = None

def parse_parameters():
    parser = argparse.ArgumentParser()
    # please adhere to alphabetical ordering
    parser.add_argument('--data_type', type=int, default=DatasetGenerator.SPIRAL,
        help='Which data set to use: (0) two circles, (1) squares, (2) two clusters, (3) spiral.')
    parser.add_argument('--dimension', type=int, default=10,
        help='Number P of samples (Y^i,X^i)^P_{i=1} to generate for the desired dataset type.')
    parser.add_argument('--noise', type=float, default=0.,
        help='Amount of noise in [0,1] to use.')
    parser.add_argument('--seed', type=int, default=None,
        help='Seed to use for random number generators.')
    parser.add_argument('--train_data_files', type=str, nargs='+', default=[],
        help='training CSV file name.')
    parser.add_argument('--train_test_ratio', type=float, default=0.5,
        help='ratio in [0,1] to split dataset into training and test part.')
    parser.add_argument('--test_data_files', type=str, nargs='+', default=[],
        help='testing CSV file name.')
    parser.add_argument('--verbose', '-v', action='count',
        help='Level of verbosity during compare')
    parser.add_argument('--version', action="store_true",
        help='Gives version information')
    return parser.parse_known_args()


def main(_):
    global FLAGS

    print("Generating input data")
    dataset_generator=DatasetGenerator()
    xs, ys = dataset_generator.generate(
        dimension=FLAGS.dimension*FLAGS.number_files,
        noise=FLAGS.noise,
        data_type=FLAGS.data_type)

    randomize = np.arange(len(xs))
    #print("Randomized set is "+str(randomize))
    np.random.shuffle(randomize)
    xs[:] = np.array(xs)[randomize]
    ys[:] = np.array(ys)[randomize]

    slice_ratio = int(FLAGS.dimension*FLAGS.train_test_ratio)
    other_slice_ratio = int(FLAGS.dimension*(1.-FLAGS.train_test_ratio))
    slice_index = slice_ratio*FLAGS.number_files

    if len(FLAGS.train_data_files) != 0:
        for l in range(FLAGS.number_files):
            logging.info("Writing to training file "+FLAGS.train_data_files[l])
            start_index = l*slice_ratio
            end_index = (l+1)*slice_ratio
            with open(FLAGS.train_data_files[l], 'w', newline='') as train_data_file:
                csv_writer = csv.writer(train_data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['x1', 'x2', 'label'])
                for x,y in zip(xs[start_index:end_index], ys[start_index:end_index]):
                    csv_writer.writerow([x[0], x[1], y[0]])
                train_data_file.close()

    if len(FLAGS.test_data_files) != 0:
        for l in range(FLAGS.number_files):
            logging.info("Writing to test file "+FLAGS.test_data_files[l])
            start_index = slice_index+l*other_slice_ratio
            end_index = slice_index+(l+1)*other_slice_ratio
            with open(FLAGS.test_data_files[l], 'w', newline='') as test_data_file:
                csv_writer = csv.writer(test_data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['x1', 'x2', 'label'])
                for x,y in zip(xs[start_index:end_index], ys[start_index:end_index]):
                    csv_writer.writerow([x[0], x[1], y[0]])
                test_data_file.close()


def internal_main():
    global FLAGS

    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    FLAGS, unparsed = parse_parameters()

    react_generally_to_options(FLAGS, unparsed)

    # init random: None will use random seed
    if FLAGS.seed is not None:
        np.random.seed(FLAGS.seed)

    # get number of files
    if len(FLAGS.train_data_files) != 0:
        FLAGS.number_files = len(FLAGS.train_data_files)
    elif len(FLAGS.test_data_files) != 0:
        FLAGS.number_files = len(FLAGS.test_data_files)
    else:
        logging.critial("Neither test nor train output filenames specified!")
        sys.exit(255)

    # check for same number of files
    if (len(FLAGS.train_data_files) != 0 and len(FLAGS.test_data_files) != 0) \
        and (len(FLAGS.train_data_files) !=len(FLAGS.test_data_files)):
        logging.critical("The same number of test and train files need to be specified.")
        sys.exit(255)

    # check dimension
    if FLAGS.dimension % FLAGS.number_files != 0:
        logging.warning("Truncating dimension to multiple of number of input files.")
    FLAGS.dimension = int(FLAGS.dimension/FLAGS.number_files)

    main([sys.argv[0]] + unparsed)