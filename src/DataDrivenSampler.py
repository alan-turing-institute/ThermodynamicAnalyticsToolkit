#!/usr/bin/env python3
#
# This a command-line version of some of the features found under
# http://playground.tensorflow.org/
# aimed at comparing steepest descent optimizer with a bayesian
# sampling approach.
#
# (C) Frederik Heber 2017-09-18

import argparse, os, sys
import tensorflow as tf
import csv
import numpy as np

from classificationdatasets import ClassificationDatasets as DatasetGenerator
from neuralnetwork import NeuralNetwork
from helpers import get_list_from_string

FLAGS = None


def main(_):
    # init random: None will use random seed
    np.random.seed(FLAGS.seed)

    print("Generating input data")
    dsgen=DatasetGenerator()
    ds = dsgen.generate(
        dimension=FLAGS.dimension,
        noise=FLAGS.noise,
        data_type=FLAGS.data_type)
    # use all as test set
    ds.set_test_train_ratio(1)

    print("Constructing neural network")
    # extract hidden layer dimensions, as "8 8" or 8 8 or whatever
    hidden_dimension=get_list_from_string(FLAGS.hidden_dimension)
    input_columns=get_list_from_string(FLAGS.input_columns)
    nn=NeuralNetwork()
    input_dimension = 2
    output_dimension = 1
    xinput, x = dsgen.create_input_layer(input_dimension, input_columns)
    nn.create(
        x,
        len(input_columns), hidden_dimension, output_dimension,
        optimizer=FLAGS.optimizer,
        seed=FLAGS.seed,
        noise_scale=FLAGS.noise_scale,
        add_dropped_layer=False)
    y_ = nn.get("y_")

    sess = tf.Session()
    nn.init_graph(sess)

    do_write_trajectory_file = False
    if FLAGS.trajectory_file is not None:
        do_write_trajectory_file = True
        no_weights = nn.get("weights").get_shape()[0]
        trajectory_file = open(FLAGS.trajectory_file, 'w', newline='')
        trajectory_writer = csv.writer(trajectory_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        trajectory_writer.writerow(['step', 'loss']+[str("weight")+str(i) for i in range(0,no_weights)])


    test_nodes = list(map(lambda key: nn.get(key), [
        "merged", "train_step", "accuracy", "train_rate", "global_step", "loss"]))+[xinput]+list(map(lambda key: nn.get(key), ["y_", "y"]))
    learning_decay = nn.get("learning_decay")
    learning_decay_power = nn.get("learning_decay_power")
    learning_rate= nn.get("learning_rate")
    print("Starting to train")
    test_intervals = max(10, FLAGS.max_steps/100)
    summary_intervals = max(20,FLAGS.max_steps/10)
    for i in range(FLAGS.max_steps):
        test_xs, test_ys = ds.get_testset()
        summary, _, acc, rate, global_step, loss_eval, xinputeval, y_true_eval, y_eval = sess.run(
            test_nodes,
            feed_dict={
                xinput: test_xs, y_: test_ys,
                learning_decay: FLAGS.learning_decay, learning_decay_power: FLAGS.learning_decay_power,
                learning_rate: FLAGS.learning_rate
        })
        if do_write_trajectory_file:
            weights_eval = sess.run(
                nn.get("weights"),
                feed_dict={
                    xinput: test_xs, y_: test_ys,
                    learning_decay: FLAGS.learning_decay, learning_decay_power: FLAGS.learning_decay_power,
                    learning_rate: FLAGS.learning_rate
            })
            trajectory_writer.writerow(
                [i, loss_eval] + [item for sublist in weights_eval for item in sublist])

        print('Accuracy at step %s (%s): %s, using rate %s' % (i, global_step, acc, rate))
        #print('Loss at step %s: %s' % (i, loss_eval))
        #print('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
        #print('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
    if do_write_trajectory_file:
        trajectory_file.close()
    print("TRAINED.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # please adhere to alphabetical ordering
    parser.add_argument('--batch_size', type=int, default=10,
        help='The number of samples used to divide sample set into batches in one training step.')
    parser.add_argument('--data_type', type=int, default=DatasetGenerator.SPIRAL,
        help='Which data set to use: (0) two circles, (1) squares, (2) two clusters, (3) spiral.')
    parser.add_argument('--dimension', type=int, default=10,
        help='Number P of samples (Y^i,X^i)^P_{i=1} to generate for the desired dataset type.')
    parser.add_argument('--dropout', type=float, default=0.9,
        help='Keep probability for training dropout, e.g. 0.9')
    parser.add_argument('--hidden_dimension', type=str, nargs='+', default=[],
        help='Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected')
    parser.add_argument('--input_columns', type=str, nargs='+', default="1 2",
        help='Pick a list of the following: (1) x1, (2) x2, (3) x1^2, (4) x2^2, (5) sin(x1), (6) sin(x2).')
    parser.add_argument('--learning_decay', type=float, default=0.001,
        help='Parameter governing the decay of the rate as lambda*t^gamma')
    parser.add_argument('--learning_decay_power', type=float, default=-.55,
        help='Parameter governing power of the decay of the rate as lambda*t^gamma, should be in -[0.5,1]')
    parser.add_argument('--learning_rate', type=float, default=0.03,
        help='Initial learning rate, e.g. 0.01')
    parser.add_argument('--log_dir', type=str, default=None,
        help='Summaries log directory')
    parser.add_argument('--max_steps', type=int, default=1000,
        help='Number of steps to run trainer.')
    parser.add_argument('--noise', type=float, default=0.,
        help='Amount of noise in [0,1] to use.')
    parser.add_argument('--noise_scale', type=float, default=1.,
        help='Relative scale of injected noise in Stochastic Gradient Langevin Dynamics compared to gradient')
    parser.add_argument('--optimizer', type=str, default="GradientDescent",
        help='Choose the optimizer to use for training: GradientDescent, StochasticGradientLangevinDynamics')
    parser.add_argument('--seed', type=int, default=None,
        help='Seed to use for random number generators.')
    parser.add_argument('--trajectory_file', type=str, default=None,
        help='CSV file name to output trajectories of sampling, i.e. weights and evaluated loss function.')
    parser.add_argument('--version', action="store_true",
        help='Gives version information')
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.version:
        # give version and exit
        print(sys.argv[0]+" version 0.1")
        sys.exit(0)

    print("Using parameters: "+str(FLAGS))
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

