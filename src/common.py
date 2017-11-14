import csv
import os
import tensorflow as tf

from DataDrivenSampler.datasets.classificationdatasets import ClassificationDatasets as DatasetGenerator
from DataDrivenSampler.models.neuralnetwork import NeuralNetwork


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


def setup_trajectory_file(filename, no_weights, no_biases, config_map):
    """ Opens the trajectory file if a proper `filename` is given.

    :param filename: filename of trajectory file or None
    :param config_map: configuration dictionary
    :return: CSV writer or None
    """
    if filename is not None:
        config_map["do_write_trajectory_file"] = True
        trajectory_writer, config_map["trajectory_file"] = \
            setup_csv_file(filename, ['step', 'loss']
                           + [str("weight")+str(i) for i in range(0,no_weights)]
                           + [str("bias") + str(i) for i in range(0, no_biases)])
        return trajectory_writer
    else:
        return None


def closeFiles(config_map):
    """ Closes the output files if they have been opened.

    :param config_map: configuration dictionary
    """
    if config_map["do_write_run_file"]:
        assert config_map["csv_file"] is not None
        config_map["csv_file"].close()
        config_map["csv_file"] = None
    if config_map["do_write_trajectory_file"]:
        assert config_map["trajectory_file"] is not None
        config_map["trajectory_file"].close()
        config_map["trajectory_file"] = None


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
    xinput, x = dsgen.create_input_layer(config_map["input_dimension"], input_columns)
    return xinput, x, ds


def construct_network_model(FLAGS, config_map, x,
                            hidden_activation=tf.nn.relu, output_activation=tf.nn.tanh,
                            loss_name="mean_squared"):
    """ Constructs the neural network

    :param FLAGS: FLAGS dictionary with command-line parameters
    :param config_map: configuration dictionary
    :param x: input layer
    :param hidden_activation: activation function for the hidden layer
    :param output_activation: activation function for the output layer
    :param loss_name: name of global loss to use
    :return: neural network
    """
    print("Constructing neural network")
    hidden_dimension=get_list_from_string(FLAGS.hidden_dimension)
    nn=NeuralNetwork()
    try:
        train_method = FLAGS.sampler
    except AttributeError:
        train_method = FLAGS.optimizer
    nn.create(
        x, hidden_dimension, config_map["output_dimension"],
        optimizer=train_method,
        seed=FLAGS.seed,
        add_dropped_layer=False,
        hidden_activation=hidden_activation,
        output_activation=output_activation,
        loss_name=loss_name
    )
    return nn


def get_activations():
    """ Returns a dictionary with all known activation functions

    :return: dictionary with activations
    """
    activations = {
        "tanh": tf.nn.tanh,
        "sigmoid": tf.nn.sigmoid,
        "softplus": tf.nn.softplus,
        "softsign": tf.nn.softsign,
        "elu": tf.nn.elu,
        "relu6": tf.nn.relu6,
        "relu": tf.nn.relu,
        "linear": tf.identity
    }
    return activations
