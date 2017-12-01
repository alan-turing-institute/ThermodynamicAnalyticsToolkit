import csv
import os
import sys
import tensorflow as tf
import pandas as pd

from DataDrivenSampler.datasets.classificationdatasets import ClassificationDatasets as DatasetGenerator
from DataDrivenSampler.version import get_package_version, get_build_hash


def get_filename_from_fullpath(fullpath):
    """ Returns the filename for any given full path

    :param fullpath: string containing filename and folders
    :return: just the filename
    """
    return os.path.basename(fullpath)

def get_list_from_string(str_or_list_of_str):
    """ Extracts list of ints from any string (or list of strings).

    :param str_or_list_of_str: string
    :return: list of ints
    """
    tmpstr=str_or_list_of_str
    if str_or_list_of_str is not str:
        try:
            tmpstr=" ".join(str_or_list_of_str)
        except(TypeError):
            tmpstr=" ".join([item for sublist in str_or_list_of_str for item in sublist])
    return [int(item) for item in tmpstr.split()]


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


def create_classification_dataset(FLAGS, config_map):
    """ Creates the dataset object using a classification generator.

    :param FLAGS: FLAGS dictionary with command-line parameters
    :param config_map: configuration dictionary
    :return: placeholder node for input, input layer for network, dataset object
    """
    print("Generating input data")
    dsgen=DatasetGenerator()
    ds = dsgen.generate(
        dimension=FLAGS.dimension,
        noise=FLAGS.noise,
        data_type=FLAGS.data_type)
    # use all as train set
    ds.set_test_train_ratio(1)
    dsgen.setup_config_map(config_map)

    # generate input layer
    input_columns = get_list_from_string(FLAGS.input_columns)
    xinput, x = create_input_layer(config_map["input_dimension"], input_columns)
    return xinput, x, ds


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
    parser.add_argument('--input_columns', type=str, nargs='+', default="1 2",
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


def create_input_pipeline(filenames, batch_size, shuffle=False, num_epochs=None, seed=None):
    """ creates a Tensorflow input pipeline given some files and
    a batch_size

    :param filenames: name of file
    :param batch_size: size of each batch to be delivered
    :param shuffle: whether to shuffle dataset or not
    :param num_epochs: number of maximum epochs, None means no limit
    :param seed: random number seed used for reshuffling
    :return: Tensorflow nodes to receive features and labels
    """
    filesize=sum([file_length(filename) for filename in filenames])
    if filesize < 1e6 and len(filenames) == 1:
        # parse file and have dataset in memory
        df = pd.read_csv(filenames[0], sep=',', header=0)
        feature_header = list(df)
        assert( 'label' in feature_header )
        feature_header.remove('label')
        features=df.loc[:,feature_header].values
        labels=df.loc[:,['label']].values #np.asarray

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.name_scope('input'):
            # Input data, pin to CPU because rest of pipeline is CPU-only
            with tf.device('/cpu:0'):
                input_features = tf.constant(features)
                input_labels = tf.constant(labels)

            feature, label = tf.train.slice_input_producer(
                [input_features, input_labels], shuffle=shuffle,
                num_epochs=num_epochs)
    else:
        filename_queue = tf.train.string_input_producer(filenames,
                                                        num_epochs=num_epochs,
                                                        shuffle=shuffle,
                                                        seed=seed)
        feature, label = read_from_csv(filename_queue)

    print("Using batch size %d" % (batch_size))
    if shuffle:
        min_after_dequeue = 30
        capacity = min_after_dequeue + 10*batch_size
        feature_batch, label_batch = tf.train.shuffle_batch(
            [feature, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue, seed=seed)
    else:
        capacity = batch_size
        feature_batch, label_batch = tf.train.batch(
            [feature, label], batch_size=batch_size, capacity=capacity)
    return feature_batch, label_batch


def create_input_layer(input_dimension, input_list):
    """ Creates the input layer of TensorFlow's neural network.

     As the input nodes are directly connected to the type of data we feed
     into the network, the function is associated with the dataset generator
     class.

     As the datasets all have two-dimensional input, several expression may
     be derived from this: first coordinate, second coordinate, squared first,
     squared second, sine of first, sine of second.

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

        # pick from the various available input columns
        arg_list_names = ["x1", "x2", "x1^2", "x2^2", "sin(x1)", "sin(x2)"]
        picked_list_names = list(map(lambda i: arg_list_names[i - 1], input_list))
        print("Picking as input columns: " + str(picked_list_names))
        arg_list = [xinput[:, 0], xinput[:, 1]]
        arg_list += [arg_list[0] * arg_list[0],
                     arg_list[1] * arg_list[1],
                     tf.sin(arg_list[0]),
                     tf.sin(arg_list[1])]
        picked_list = list(map(lambda i: arg_list[i - 1], input_list))
        x = tf.transpose(tf.stack(picked_list))
        print("x is " + str(x.get_shape()))
    return xinput, x

