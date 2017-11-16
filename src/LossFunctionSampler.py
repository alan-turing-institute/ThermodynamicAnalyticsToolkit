#!/usr/bin/env @PYTHON@

import sys, getopt
#sys.path.insert(1, '@pythondir@')

import tensorflow as tf

import numpy as np

import argparse

from DataDrivenSampler.datasets.classificationdatasets import ClassificationDatasets

from DataDrivenSampler.common import get_filename_from_fullpath, setup_csv_file
from DataDrivenSampler.models.model import model
from DataDrivenSampler.version import get_package_version, get_build_hash

FLAGS = None

def parse_parameters():
    """ Sets up the argument parser for parsing command line parameters into dictionary

    :return: dictionary with parameter names as keys, unrecognized parameters
    """
    parser = argparse.ArgumentParser()
    # please adhere to alphabetical ordering
    parser.add_argument('--csv_file', type=str, default=None,
        help='CSV file name to output sampled values to.')
    parser.add_argument('--data_type', type=int, default=ClassificationDatasets.SPIRAL,
        help='Which data set to use: (0) two circles, (1) squares, (2) two clusters, (3) spiral.')
    parser.add_argument('--dimension', type=int, default=10,
        help='Number P of samples (Y^i,X^i)^P_{i=1} to generate for the desired dataset type.')
    parser.add_argument('--hidden_activation', type=str, default="relu",
        help='Activation function to use for hidden layer: tanh, relu, linear')
    parser.add_argument('--hidden_dimension', type=str, nargs='+', default=[],
        help='Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected')
    parser.add_argument('--input_columns', type=str, nargs='+', default="1 2",
        help='Pick a list of the following: (1) x1, (2) x2, (3) x1^2, (4) x2^2, (5) sin(x1), (6) sin(x2).')
    parser.add_argument('--interval_biases', type=str, nargs='+', default=[],
        help='Min and max value for each bias.')
    parser.add_argument('--interval_weights', type=str, nargs='+', default=[],
        help='Min and max value for each weight.')
    parser.add_argument('--loss', type=str, default="mean_squared",
        help='Set the loss to be measured during sampling, e.g. mean_squared, log_loss, ...')
    parser.add_argument('--noise', type=float, default=0.,
        help='Amount of noise in [0,1] to use.')
    parser.add_argument('--output_activation', type=str, default="tanh",
        help='Activation function to use for output layer: tanh, relu, linear')
    parser.add_argument('--samples_biases', type=int, default=None,
        help='Number of samples to take per bias interval')
    parser.add_argument('--samples_weights', type=int, default=None,
        help='Number of samples to take per weight interval')
    parser.add_argument('--seed', type=int, default=None,
        help='Seed to use for random number generators.')
    parser.add_argument('--version', '-V', action="store_true",
        help='Gives version information')
    return parser.parse_known_args()

def main(_):
    network_model = model(FLAGS)

    network_model.init_network(None, setup=None)

    sess = network_model.sess
    weights = tf.get_collection(tf.GraphKeys.WEIGHTS)
    weights_placeholder = tf.placeholder(shape=weights[0].get_shape(), dtype=weights[0].dtype.base_dtype)
    set_weights_t = weights[0].assign(weights_placeholder)
    biases = tf.get_collection(tf.GraphKeys.BIASES)
    biases_placeholder = tf.placeholder(shape=biases[0].get_shape(), dtype=biases[0].dtype.base_dtype)
    set_biases_t = biases[0].assign(biases_placeholder)
    batch_xs, batch_ys = network_model.ds.next_batch(FLAGS.dimension)
    feed_dict = {
        network_model.xinput: batch_xs,
        network_model.nn.placeholder_nodes["y_"]: batch_ys
    }
    loss = network_model.nn.get_list_of_nodes(["loss"])

    if FLAGS.csv_file is not None:
        csv_writer, csv_file = setup_csv_file(FLAGS.csv_file, ["w1","w2","b","loss"])

    weights_interval_start = float(FLAGS.interval_weights[0])
    weights_interval_end = float(FLAGS.interval_weights[1])
    weights_interval_length = weights_interval_end - weights_interval_start
    biases_interval_start = float(FLAGS.interval_biases[0])
    biases_interval_end = float(FLAGS.interval_biases[1])
    biases_interval_length = biases_interval_end - biases_interval_start
    for w1 in np.arange(0,FLAGS.samples_weights+1)*weights_interval_length/float(FLAGS.samples_weights+1)+weights_interval_start:
        for w2 in np.arange(0, FLAGS.samples_weights + 1) * weights_interval_length / float(FLAGS.samples_weights + 1)+weights_interval_start:
            for b in np.arange(0, FLAGS.samples_biases + 1) * biases_interval_length / float(
                            FLAGS.samples_biases + 1)+biases_interval_start:
                # set the parameters
                sess.run([set_weights_t, set_biases_t], feed_dict={
                    weights_placeholder: [[w1], [w2]],
                    biases_placeholder: [b]
                })

                # evaluate the loss
                loss_eval = sess.run(loss, feed_dict=feed_dict)
                print("Loss at the given parameters w("+str(w1)+","+str(w2)+"), b("
                      +str(b)+") is "+str(loss_eval[0]))

                if FLAGS.csv_file is not None:
                    csv_writer.writerow([w1,w2,b,loss_eval[0]])

    network_model.finish()

    if FLAGS.csv_file is not None:
        csv_file.close()

if __name__ == '__main__':
    FLAGS, unparsed = parse_parameters()

    if FLAGS.version:
        # give version and exit
        print(get_filename_from_fullpath(sys.argv[0])+" "+get_package_version()+" -- version "+get_build_hash())
        sys.exit(0)

    print("Using parameters: "+str(FLAGS))

    if len(unparsed) != 0:
        print("There are unparsed parameters '"+str(unparsed)+"', have you misspelled some?")
        sys.exit(255)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


