#!/usr/bin/env python3
#
# This a command-line version of some of the features found under
# http://playground.tensorflow.org/
# aimed at comparing steepest descent optimizer with a bayesian
# sampling approach.
#
# (C) Frederik Heber 2017-09-18

import argparse
import csv
import numpy as np
import sys

import tensorflow as tf
from models.helpers import get_list_from_string

from datasets.classificationdatasets import ClassificationDatasets as DatasetGenerator
from models.neuralnetwork import NeuralNetwork

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
        add_dropped_layer=False)
    y_ = nn.get("y_")
    step_width = nn.get("step_width")
    assert step_width is not None

    sess = tf.Session()
    nn.init_graph(sess)

    do_write_csv_file = False
    if FLAGS.csv_file is not None:
        do_write_csv_file = True
        csv_file = open(FLAGS.csv_file, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['step', 'epoch', 'accuracy', 'loss', 'scaled_gradient'])

    do_write_trajectory_file = False
    if FLAGS.trajectory_file is not None:
        do_write_trajectory_file = True
        no_weights = nn.get("weights").get_shape()[0]
        no_biases = nn.get("biases").get_shape()[0]
        trajectory_file = open(FLAGS.trajectory_file, 'w', newline='')
        trajectory_writer = csv.writer(trajectory_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        trajectory_writer.writerow(['step', 'loss']
                                   +[str("weight")+str(i) for i in range(0,no_weights)]
                                   +[str("bias") + str(i) for i in range(0, no_biases)])

    test_nodes = list(map(lambda key: nn.get(key), [
        "merged", "train_step", "accuracy", "global_step", "loss", "y_", "y", "scaled_gradient"]))
    assert None not in test_nodes
    print("Starting to train")
    for i in range(FLAGS.max_steps):
        print("Current step is "+str(i))
        test_xs, test_ys = ds.get_testset()
        assert not any(item is None for item in [test_xs, test_ys])
        # print("Testset is x: "+str(test_xs[0:5])+", y: "+str(test_ys[0:5]))
        summary, _, acc, global_step, loss_eval, y_true_eval, y_eval, scaled_grad = sess.run(
            test_nodes,
            feed_dict={
                xinput: test_xs, y_: test_ys,
                step_width: FLAGS.step_width
        })

        if i % FLAGS.every_nth == 0:
            if do_write_csv_file:
                csv_writer.writerow([global_step, i, acc, loss_eval, scaled_grad])
            if do_write_trajectory_file:
                weights_eval, biases_eval = sess.run(
                    [nn.get("weights"), nn.get("biases")],
                    feed_dict={
                        xinput: test_xs, y_: test_ys,
                        step_width: FLAGS.step_width
                    })
                trajectory_writer.writerow(
                    [i, loss_eval]
                    + [item for sublist in weights_eval for item in sublist]
                    + [item for item in biases_eval])

        print('Accuracy at step %s (%s): %s' % (i, global_step, acc))
        #print('Loss at step %s: %s' % (i, loss_eval))
        #print('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
        #print('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
    if do_write_csv_file:
        csv_file.close()
    if do_write_trajectory_file:
        trajectory_file.close()
    print("TRAINED.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # please adhere to alphabetical ordering
    parser.add_argument('--csv_file', type=str, default=None,
        help='CSV run file name to output accuracy and loss values.')
    parser.add_argument('--data_type', type=int, default=DatasetGenerator.SPIRAL,
        help='Which data set to use: (0) two circles, (1) squares, (2) two clusters, (3) spiral.')
    parser.add_argument('--dimension', type=int, default=10,
        help='Number P of samples (Y^i,X^i)^P_{i=1} to generate for the desired dataset type.')
    parser.add_argument('--dropout', type=float, default=0.9,
        help='Keep probability for training dropout, e.g. 0.9')
    parser.add_argument('--every_nth', type=int, default=1,
        help='Store only every nth trajectory (and run) point to files, e.g. 10')
    parser.add_argument('--hidden_dimension', type=str, nargs='+', default=[],
        help='Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected')
    parser.add_argument('--input_columns', type=str, nargs='+', default="1 2",
        help='Pick a list of the following: (1) x1, (2) x2, (3) x1^2, (4) x2^2, (5) sin(x1), (6) sin(x2).')
    parser.add_argument('--max_steps', type=int, default=1000,
        help='Number of steps to run trainer.')
    parser.add_argument('--noise', type=float, default=0.,
        help='Amount of noise in [0,1] to use.')
    parser.add_argument('--optimizer', type=str, default="GradientDescent",
        help='Choose the optimizer to use for sampling: GradientDescent')
    parser.add_argument('--seed', type=int, default=None,
        help='Seed to use for random number generators.')
    parser.add_argument('--step_width', type=float, default=0.03,
        help='step width \Delta t to use, e.g. 0.01')
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

