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

from classification_datasets import classification_datasets as dataset_generator
from neuralnetwork import neuralnetwork

FLAGS = None


def create_input_layer(input_dimension, input_list):
    # Input placeholders
    with tf.name_scope('input'):
        xinput = tf.placeholder(tf.float32, [None, input_dimension], name='x-input')
        #print("xinput is "+str(xinput.get_shape()))

        # pick from the various available input columns
        arg_list_names= ["x1", "x2", "x1^2", "x2^2", "sin(x1)", "sin(x2)"]
        picked_list_names = list(map(lambda i:arg_list_names[i-1], input_list))
        print("Picking as input columns: "+str(picked_list_names))
        arg_list = [ xinput[:,0], xinput[:,1] ]
        arg_list += [arg_list[0]*arg_list[0],
                     arg_list[1]*arg_list[1],
                     tf.sin(arg_list[0]),
                     tf.sin(arg_list[1])]
        picked_list = list(map(lambda i:arg_list[i-1], input_list))
        x = tf.transpose(tf.stack(picked_list))
        print("x is "+str(x.get_shape()))
    return xinput, x


def main(_):
    # init random: None will use random seed
    np.random.seed(FLAGS.seed)

    print("Generating input data")
    dsgen=dataset_generator()
    ds = dsgen.generate(
        dimension=FLAGS.dimension,
        noise=FLAGS.noise,
        data_type=FLAGS.data_type)

    do_write_summaries = False
    if FLAGS.log_dir is not None:
        do_write_summaries = True

    print("Constructing neural network")
    nn=neuralnetwork()
    input_dimension = 2
    output_dimension = 1
    xinput, x = create_input_layer(input_dimension, FLAGS.input_columns)
    nn.create(
        x,
        len(FLAGS.input_columns), FLAGS.hidden_dimension, output_dimension,
        optimizer=FLAGS.optimizer,
        seed=FLAGS.seed,
        noise_scale=FLAGS.noise_scale)
    y_ = nn.get("y_")
    keep_prob = nn.get("keep_prob")

    print("Starting session")
    sess = tf.Session()
    if do_write_summaries:
        nn.add_writers(sess, FLAGS.log_dir)
    nn.init_graph(sess)
    if do_write_summaries:
        nn.add_graphing_train()
        nn.graph_truth(sess, ds.xs, ds.ys, FLAGS.dimension)

    do_write_csv_file = False
    if FLAGS.csv_file is not None:
        do_write_csv_file = True
        csv_file = open(FLAGS.csv_file, 'w', newline='')
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        if FLAGS.optimizer == "StochasticGradientLangevinDynamics":
            csv_writer.writerow(['step', 'epoch', 'accuracy', 'loss', 'rate', 'noise'])
        else:
            csv_writer.writerow(['step', 'epoch', 'accuracy', 'loss', 'rate'])

    do_write_trajectory_file = False
    if FLAGS.trajectory_file is not None:
        do_write_trajectory_file = True
        no_weights = nn.get("weights").get_shape()[0]
        trajectory_file = open(FLAGS.trajectory_file, 'w', newline='')
        trajectory_writer = csv.writer(trajectory_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        trajectory_writer.writerow(['step', 'loss']+[str("weight")+str(i) for i in range(0,no_weights)])


    test_nodes = list(map(lambda key: nn.get(key), [
        "merged", "accuracy", "train_rate", "global_step", "loss"]))+[xinput]+list(map(lambda key: nn.get(key), ["y_", "y"]))
    train_nodes = list(map(lambda key: nn.get(key), [
        "merged", "train_step"]))
    learning_decay = nn.get("learning_decay")
    learning_decay_power = nn.get("learning_decay_power")
    learning_rate= nn.get("learning_rate")
    if do_write_summaries:
        train_writer = nn.get("train_writer")
        test_writer = nn.get("test_writer")
    print("Starting to train")
    test_intervals = max(10, FLAGS.max_steps/100)
    summary_intervals = max(20,FLAGS.max_steps/10)
    for i in range(FLAGS.max_steps):
        #print("Current training step is %d" % i)
        if (i % test_intervals == 0):  # test
            test_xs, test_ys = ds.get_testset()
            summary, acc, rate, global_step, loss_eval, xinputeval, y_true_eval, y_eval = sess.run(
                test_nodes,
                feed_dict={
                    xinput: test_xs, y_: test_ys,
                    keep_prob: 1.,
                    learning_decay: FLAGS.learning_decay, learning_decay_power: FLAGS.learning_decay_power,
                    learning_rate: FLAGS.learning_rate
            })
            if do_write_trajectory_file:
                weights_eval = sess.run(
                    nn.get("weights"),
                    feed_dict={
                        xinput: test_xs, y_: test_ys,
                        keep_prob: 1.,
                        learning_decay: FLAGS.learning_decay, learning_decay_power: FLAGS.learning_decay_power,
                        learning_rate: FLAGS.learning_rate
                })
                trajectory_writer.writerow(
                    [i, loss_eval] + [item for sublist in weights_eval for item in sublist])
            if do_write_csv_file:
                if FLAGS.optimizer == "StochasticGradientLangevinDynamics":
                    noise = sess.run(
                    nn.get("random_noise"),
                    feed_dict={
                        xinput: test_xs, y_: test_ys,
                        keep_prob: 1.,
                        learning_decay: FLAGS.learning_decay, learning_decay_power: FLAGS.learning_decay_power,
                        learning_rate: FLAGS.learning_rate
                    })
                    csv_writer.writerow([global_step, i, acc, loss_eval, rate, noise])
                else:
                    csv_writer.writerow([global_step, i, acc, loss_eval, rate])
            if do_write_summaries:
                test_writer.add_summary(summary, i)
                plot_buf = nn.get_plot_buf(xinputeval, y_eval, FLAGS.dimension)
                plot_image_summary_ = sess.run(
                    nn.get("plot_image_summary_test"),
                    feed_dict={nn.get("plot_buf_test"): plot_buf.getvalue()})
                test_writer.add_summary(plot_image_summary_, global_step=i)

            print('Accuracy at step %s (%s): %s, using rate %s' % (i, global_step, acc, rate))
            #print('Loss at step %s: %s' % (i, loss_eval))
            #print('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
            #print('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
        else:  # Record train set summaries, and train
            while not ds.epochFinished():
                batch_xs, batch_ys = ds.next_batch(FLAGS.batch_size)
                #print("Current batch is: "+str(batch_xs[0:9].transpose())+" --- "+str(batch_ys[0:9].transpose()))
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                if i % summary_intervals == summary_intervals-1:  # Record execution stats
                    summary, _ = sess.run(
                        train_nodes,
                        feed_dict={
                            xinput: batch_xs, y_: batch_ys,
                            keep_prob: FLAGS.dropout,
                            learning_decay: FLAGS.learning_decay, learning_decay_power: FLAGS.learning_decay_power,
                            learning_rate: FLAGS.learning_rate},
                        options=run_options,
                        run_metadata=run_metadata)
                else:  # train
                    summary, _ = sess.run(
                        train_nodes,
                        feed_dict={
                            xinput: batch_xs, y_: batch_ys,
                            keep_prob: FLAGS.dropout,
                            learning_decay: FLAGS.learning_decay, learning_decay_power: FLAGS.learning_decay_power,
                            learning_rate: FLAGS.learning_rate})
            #print("Epoch done.")
            ds.resetEpoch()
            if do_write_summaries:
                if i % summary_intervals == summary_intervals-1:  # Record execution stats
                    train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                    print('Adding run metadata for', i)
                train_writer.add_summary(summary, global_step=i)
    if do_write_csv_file:
        csv_file.close()
    if do_write_trajectory_file:
        trajectory_file.close()
    if do_write_summaries:
        train_writer.close()
        test_writer.close()
    print("TRAINED.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # please adhere to alphabetical ordering
    parser.add_argument('--batch_size', type=int, default=10,
        help='The number of samples used to divide sample set into batches in one training step.')
    parser.add_argument('--csv_file', type=str, default=None,
        help='CSV file name to output accuracy and loss values.')
    parser.add_argument('--data_type', type=int, default=dataset_generator.SPIRAL,
        help='Which data set to use: (0) two circles, (1) squares, (2) two clusters, (3) spiral.')
    parser.add_argument('--dimension', type=int, default=10,
        help='Number P of samples (Y^i,X^i)^P_{i=1} to generate for the desired dataset type.')
    parser.add_argument('--dropout', type=float, default=0.9,
        help='Keep probability for training dropout, e.g. 0.9')
    parser.add_argument('--hidden_dimension', type=int, nargs='+', default=[],
        help='Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected')
    parser.add_argument('--input_columns', type=int, nargs='+', default=[1, 2],
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
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

