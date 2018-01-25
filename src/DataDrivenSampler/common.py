import argparse
import collections
import csv
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf

from DataDrivenSampler.datasets.classificationdatasets import ClassificationDatasets as DatasetGenerator
from DataDrivenSampler.version import get_package_version, get_build_hash


def get_filename_from_fullpath(fullpath):
    """ Returns the filename for any given full path

    :param fullpath: string containing filename and folders
    :return: just the filename
    """
    return os.path.basename(fullpath)

def get_list_from_string(str_or_list_of_str):
    """ Extracts list of strings from any string (or list of strings).

    :param str_or_list_of_str: string
    :return: list of str
    """
    tmpstr=str_or_list_of_str
    if str_or_list_of_str is not str:
        try:
            tmpstr=" ".join(str_or_list_of_str)
        except(TypeError):
            tmpstr=" ".join([item for sublist in str_or_list_of_str for item in sublist])
    return [item for item in tmpstr.split()]


def initialize_config_map():
    """ This initialize the configuration dictionary with default values

    :return:
    """
    # output files
    config_map = {
        "do_write_run_file": False,
        "csv_file": None,
        "do_write_trajectory_file": False,
        "trajectory_file": None
    }

    return config_map


def setup_csv_file(filename, header):
    """ Opens a new CSV file and writes the given `header` to it.

    :param filename: filename of CSV file
    :param header: header to write as first row
    :return: csv writer, csv file
    """
    csv_file = open(filename, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(header)
    return csv_writer, csv_file


def setup_run_file(filename, header, config_map):
    """ Opens the run CSV file if a proper `filename` is given.

    :param filename: filename of run CSV file or None
    :param header: list of strings as header for each column
    :param config_map: configuration dictionary
    :return: CSV writer or None
    """
    if filename is not None:
        config_map["do_write_run_file"] = True
        csv_writer, config_map["csv_file"] = setup_csv_file(filename, header)
        return csv_writer
    else:
        return None


def get_trajectory_header(no_weights, no_biases):
    """ Returns the header for CSV trajectory file based on the given number
    of weights and biases.

    :param no_weights: number of weights of the network
    :param no_biases: number of biases of the network
    :return: list of strings with column names
    """
    return ['step', 'loss']\
           + [str("weight")+str(i) for i in range(0,no_weights)]\
           + [str("bias") + str(i) for i in range(0, no_biases)]


def setup_trajectory_file(filename, no_weights, no_biases, config_map):
    """ Opens the trajectory file if a proper `filename` is given.

    :param filename: filename of trajectory file or None
    :param config_map: configuration dictionary
    :return: CSV writer or None
    """
    if filename is not None:
        config_map["do_write_trajectory_file"] = True
        trajectory_writer, config_map["trajectory_file"] = \
            setup_csv_file(filename, get_trajectory_header(no_weights, no_biases))
        return trajectory_writer
    else:
        return None

def str2bool(v):
    # this is the answer from stackoverflow https://stackoverflow.com/a/43357954/1967646
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_data_options_to_parser(parser):
    """ Adding options common to both sampler and optimizer to argparse
    object for specifying the data set.

    :param parser: argparse's parser object
    """
    # please adhere to alphabetical ordering
    parser.add_argument('--batch_data_files', type=str, nargs='+', default=[],
        help='Names of CSV files to parse input data from')


def add_model_options_to_parser(parser):
    """ Adding options common to both sampler and optimizer to argparse
    object for specifying the model.

    :param parser: argparse's parser object
    """
    # please adhere to alphabetical ordering
    parser.add_argument('--batch_size', type=int, default=None,
        help='The number of samples used to divide sample set into batches in one training step.')
    parser.add_argument('--dropout', type=float, default=None,
        help='Keep probability for training dropout, e.g. 0.9')
    parser.add_argument('--fix_parameters', type=str, default=None,
        help='Fix parameters for sampling/training by stating "name=value;..."')
    parser.add_argument('--hidden_activation', type=str, default="relu",
        help='Activation function to use for hidden layer: tanh, relu, linear')
    parser.add_argument('--hidden_dimension', type=str, nargs='+', default=[],
        help='Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected')
    parser.add_argument('--in_memory_pipeline', type=str2bool, default=True,
        help='Whether to use an in-memory input pipeline (for small datasets) or the tf.Dataset module.')
    parser.add_argument('--input_columns', type=str, nargs='+', default="",
        help='Pick a list of the following: (1) x1, (2) x2, (3) x1^2, (4) x2^2, (5) sin(x1), (6) sin(x2).')
    parser.add_argument('--loss', type=str, default="mean_squared",
        help='Set the loss to be measured during sampling, e.g. mean_squared, log_loss, ...')
    parser.add_argument('--output_activation', type=str, default="tanh",
        help='Activation function to use for output layer: tanh, relu, linear')
    parser.add_argument('--seed', type=int, default=None,
        help='Seed to use for random number generators.')


def add_common_options_to_parser(parser):
    """ Adding options common to both sampler and optimizer to argparse
    object for specifying files and how to write them.

    :param parser: argparse's parser object
    """
    # please adhere to alphabetical ordering
    parser.add_argument('--every_nth', type=int, default=1,
        help='Store only every nth trajectory (and run) point to files, e.g. 10')
    parser.add_argument('--inter_ops_threads', type=int, default=1,
        help='Sets the number of threads to split up ops in between. NOTE: This hurts reproducibility to some extent because of parallelism.')
    parser.add_argument('--intra_ops_threads', type=int, default=None,
        help='Sets the number of threads to use within an op, i.e. Eigen threads for linear algebra routines.')
    parser.add_argument('--restore_model', type=str, default=None,
        help='Restore model (weights and biases) from a file.')
    parser.add_argument('--run_file', type=str, default=None,
        help='CSV run file name to runtime information such as output accuracy and loss values.')
    parser.add_argument('--save_model', type=str, default=None,
        help='Save model (weights and biases) to a file for later restoring.')
    parser.add_argument('--sql_db', type=str, default=None,
        help='Supply file for writing timing information to sqlite database')
    parser.add_argument('--trajectory_file', type=str, default=None,
        help='CSV file name to output trajectories of sampling, i.e. weights and evaluated loss function.')
    parser.add_argument('--version', '-V', action="store_true",
        help='Gives version information')


def react_to_common_options(FLAGS, unparsed):
    """ Extracted behavior for options shared between sampler and optimizer
    here for convenience.

    :param FLAGS: parsed cmd-line options as produced by argparse.parse_known_args()
    :param unparsed: unparsed cmd-line options as produced by argparse.parse_known_args()
    """
    if FLAGS.version:
        # give version and exit
        print(get_filename_from_fullpath(sys.argv[0])+" "+get_package_version()+" -- version "+get_build_hash())
        sys.exit(0)

    print("Using parameters: "+str(FLAGS))

    if len(unparsed) != 0:
        print("There are unparsed parameters '"+str(unparsed)+"', have you misspelled some?")
        sys.exit(255)


def file_length(filename):
    """ Determines the length of the file designated by `filename`.

    :param filename: name of file
    :return: length
    """
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def number_lines_in_file(filename):
    """ Determines the lines in the file designated by `filename`.

    :param filename: name of file
    :return: number of, lines
    """
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return l


def read_from_csv(filename_queue):
    """ Reads a set of records/data from a CSV file into a tensorflow tensor.

    :param filename_queue: filename
    :return: features and labels (i.e. x,y)
    """
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[0.], [0.], [0]]
    col_x1, col_x2, col_label = tf.decode_csv(
        csv_row, record_defaults=record_defaults)
    features = tf.stack([col_x1, col_x2])
    label = tf.stack([col_label])
    return features, label


def decode_csv_line(line, defaults):
    """Convert a csv line into a (features_dict,label) pair."""
    # Decode the line to a tuple of items based on the types of
    # csv_header.values().
    items = tf.decode_csv(line, list(defaults.values()))

    # reshape into proper tensors
    features = items[0:-1]
    label = tf.reshape(tf.convert_to_tensor(items[-1], dtype=tf.int32), [1])

    # return last element as label, rest as features
    return features, label


def get_csv_defaults(input_dimension, output_dimension=1):
    """ Return the defaults for a csv line with input features and output labels.

    :param input_dimension: number of features
    :param output_dimension: number of labels
    """
    defaults = collections.OrderedDict([])
    for i in range(input_dimension):
        defaults.update({"x"+str(i+1): [0.]})
    if output_dimension > 1:
        for i in range(input_dimension):
            defaults.update({"label"+str(i+1): [0]})
    else:
        defaults.update({"label": [0]})
    return defaults


def create_input_layer(input_dimension, input_list):
    """ Creates the input layer of TensorFlow's neural network.

    As the input nodes are directly connected to the type of data we feed
     into the network, the function is associated with the dataset generator
     class.

    For arbitrary input dimension we support taking powers, sine or cosine
     of the argument.

    All data resides in the domain [-r,r]^2.

    :param input_dimension: number of nodes for the input layer
    :param input_list: Pick of derived arguments to
            actually feed into the net
    :returns: generated nodes for direct input and derived input
    """
    # Input placeholders
    with tf.name_scope('input'):
        xinput = tf.placeholder(tf.float64, [None, input_dimension], name='x-input')
        # print("xinput is "+str(xinput.get_shape()))

        # parse input columns
        picked_list = []
        for token in input_list:
            # get the argument
            x_index = token.find('x')
            if x_index != -1:
                arg_name = None
                for i in range(x_index, len(token)):
                    if (token[i] < "0") or (token[i] > "9"):
                        arg_name = token[x_index:i]
                        break
                assert( arg_name is not None )
                arg = xinput[:, (int(arg_name[1:])-1)]
                if "sin" in token:
                    picked_list.append(tf.sin(arg))
                elif "cos" in token:
                    picked_list.append(tf.cos(arg))
                elif "^" in token:
                    power = int(token[(token.find('^')+1):])
                    picked_list.append(tf.pow(arg, power))
                else:
                    picked_list.append(arg)
            else:
                picked_list.append(xinput[:, (int(token) - 1)])
        # if no specific input columns are desired, take all
        if len(input_list) == 0:
            x = tf.identity(xinput)
        else:
            x = tf.transpose(tf.stack(picked_list))
        print("x is " + str(x.get_shape()))
    return xinput, x

