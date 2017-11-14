#!/usr/bin/env python3
#
# This a command-line version of some of the features found under
# http://playground.tensorflow.org/
# aimed at comparing steepest descent optimizer with a bayesian
# sampling approach.
#
# (C) Frederik Heber 2017-09-18

import argparse

import tensorflow as tf

from DataDrivenSampler.datasets.classificationdatasets import ClassificationDatasets as DatasetGenerator
from DataDrivenSampler.common import setup_run_file, setup_trajectory_file

from math import sqrt

def parse_parameters():
    """ Sets up the argument parser for parsing command line parameters into dictionary

    :return: dictionary with parameter names as keys, unrecognized parameters
    """
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
    parser.add_argument('--every_nth', type=int, default=1,
        help='Store only every nth trajectory (and run) point to files, e.g. 10')
    parser.add_argument('--friction_constant', type=float, default=0.,
        help='friction to scale the influence of momenta')
    parser.add_argument('--hidden_activation', type=str, default="relu",
        help='Activation function to use for hidden layer: tanh, relu, linear')
    parser.add_argument('--hidden_dimension', type=str, nargs='+', default=[],
        help='Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected')
    parser.add_argument('--input_columns', type=str, nargs='+', default="1 2",
        help='Pick a list of the following: (1) x1, (2) x2, (3) x1^2, (4) x2^2, (5) sin(x1), (6) sin(x2).')
    parser.add_argument('--inverse_temperature', type=float, default=1.,
        help='Inverse temperature that scales the gradients')
    parser.add_argument('--loss', type=str, default="mean_squared",
        help='Set the loss to be measured during sampling, e.g. mean_squared, log_loss, ...')
    parser.add_argument('--max_steps', type=int, default=1000,
        help='Number of steps to run trainer.')
    parser.add_argument('--noise', type=float, default=0.,
        help='Amount of noise in [0,1] to use.')
    parser.add_argument('--output_activation', type=str, default="tanh",
        help='Activation function to use for output layer: tanh, relu, linear')
    parser.add_argument('--restore_model', type=str, default=None,
        help='Restore model (weights and biases) from a file.')
    parser.add_argument('--run_file', type=str, default=None,
        help='CSV file name to output run time information such as accuracy and loss values.')
    parser.add_argument('--sampler', type=str, default="GeometricLangevinAlgorithm_1stOrder",
        help='Choose the sampler to use for sampling: GeometricLangevinAlgorithm_1stOrder, GeometricLangevinAlgorithm_2ndOrder, StochasticGradientLangevinDynamics')
    parser.add_argument('--save_model', type=str, default=None,
        help='Save model (weights and biases) to a file for later restoring.')
    parser.add_argument('--seed', type=int, default=None,
        help='Seed to use for random number generators.')
    parser.add_argument('--step_width', type=float, default=0.03,
        help='step width \Delta t to use, e.g. 0.01')
    parser.add_argument('--trajectory_file', type=str, default=None,
        help='CSV file name to output trajectories of sampling, i.e. weights and evaluated loss function.')
    parser.add_argument('--version', '-V', action="store_true",
        help='Gives version information')
    return parser.parse_known_args()


def sample(FLAGS, ds, sess, nn, xinput, run_writer, trajectory_writer, config_map):
    """ Performs the actual sampling of the neural network `nn` given a dataset `ds` and a
    Session `session`.

    :param FLAGS: FLAGS dictionary with command-line parameters
    :param ds: dataset
    :param sess: Session object
    :param nn: neural network
    :param xinput: input nodes of neural network
    :param run_writer: run csv writer
    :param trajectory_writer: trajectory csv writer
    :param config_map: configuration dictionary
    """
    # create global variable to hold kinetic energy
    with tf.variable_scope("accumulate", reuse=True):
        kinetic_energy_t = tf.get_variable("kinetic")
        zero_kinetic_energy = kinetic_energy_t.assign(0.)
        momenta_t = tf.get_variable("momenta")
        zero_momenta = momenta_t.assign(0.)
        gradients_t = tf.get_variable("gradients")
        zero_gradients = gradients_t.assign(0.)
        noise_t = tf.get_variable("noise")
        zero_noise = noise_t.assign(0.)

    placeholder_nodes = nn.get_dict_of_nodes(
        ["friction_constant", "inverse_temperature", "step_width", "y_"])
    test_nodes = nn.get_list_of_nodes(["merged", "train_step", "accuracy", "global_step", "loss", "y_", "y"])

    output_width=8
    output_precision=8

    # check that sampler's parameters are actually used
    if FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder", "GeometricLangevinAlgorithm_2ndOrder"]:
        gamma, beta, deltat = sess.run(nn.get_list_of_nodes(
            ["friction_constant", "inverse_temperature", "step_width"]), feed_dict={
                placeholder_nodes["step_width"]: FLAGS.step_width,
                placeholder_nodes["inverse_temperature"]: FLAGS.inverse_temperature,
                placeholder_nodes["friction_constant"]: FLAGS.friction_constant
            })
        print("Sampler parameters: gamma = %lg, beta = %lg, delta t = %lg" % (gamma, beta, deltat))

    print("Starting to sample")
    print_intervals = max(1,int(FLAGS.max_steps/100))
    for i in range(FLAGS.max_steps):
        #print("Current step is "+str(i))
        batch_xs, batch_ys = ds.next_batch(FLAGS.batch_size)
        feed_dict={
            xinput: batch_xs, placeholder_nodes["y_"]: batch_ys,
            placeholder_nodes["step_width"]: FLAGS.step_width,
            placeholder_nodes["inverse_temperature"]: FLAGS.inverse_temperature,
            placeholder_nodes["friction_constant"]: FLAGS.friction_constant
        }
        # print("Testset is x: "+str(test_xs[:])+", y: "+str(test_ys[:]))
        # print("Testset is x: "+str(test_xs[:])+", y: "+str(test_ys[:]))

        # zero kinetic energy
        if FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder", "GeometricLangevinAlgorithm_2ndOrder"]:
            check_kinetic, check_momenta, check_gradients, check_noise = \
                sess.run([zero_kinetic_energy, zero_momenta, zero_gradients, zero_noise])
            assert( abs(check_kinetic) < 1e-10)
            assert( abs(check_momenta) < 1e-10)
            assert( abs(check_gradients) < 1e-10)
            assert( abs(check_noise) < 1e-10)
        # NOTE: All values from nodes contained in the same call to tf.run() with train_step
        # will be evaluated as if before train_step. Nodes that are changed in the update due to
        # train_step (e.g. momentum_t) however are updated.
        # In other words, whether we use
        #   tf.run([train_step, loss_eval], ...)
        # or
        #   tf.run([loss_eval, train_step], ...)
        # is not important. Only a subsequent, distinct tf.run() call would produce a different loss_eval.
        summary, _, acc, global_step, loss_eval, y_true_eval, y_eval = \
            sess.run(test_nodes, feed_dict=feed_dict)
        if FLAGS.sampler in ["StochasticGradientLangevinDynamics", "GeometricLangevinAlgorithm_1stOrder",
                             "GeometricLangevinAlgorithm_2ndOrder"]:
            if FLAGS.sampler == "StochasticGradientLangevinDynamics":
                gradients, noise = \
                    sess.run([gradients_t, noise_t])
            else:
                kinetic_energy, momenta, gradients, noise = \
                    sess.run([kinetic_energy_t, momenta_t, gradients_t, noise_t])
        if i % FLAGS.every_nth == 0:
            if config_map["do_write_trajectory_file"]:
                weights_eval, biases_eval = sess.run(
                    [nn.get("weights"), nn.get("biases")],
                    feed_dict=feed_dict)
                trajectory_writer.writerow(
                    [global_step, loss_eval]
                    + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                       for sublist in weights_eval for item in sublist]
                    + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                       for item in biases_eval])

            if config_map["do_write_run_file"]:
                if FLAGS.sampler in ["StochasticGradientLangevinDynamics", "GeometricLangevinAlgorithm_1stOrder",
                                     "GeometricLangevinAlgorithm_2ndOrder"]:
                    if FLAGS.sampler == "StochasticGradientLangevinDynamics":
                        run_writer.writerow([global_step, i, acc, loss_eval]
                                            + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                                precision=output_precision)
                                               for x in [sqrt(gradients), sqrt(noise)]])
                    else:
                        run_writer.writerow([global_step, i, acc, loss_eval]
                                      + ['{:{width}.{precision}e}'.format(loss_eval+kinetic_energy,
                                                                          width=output_width,
                                                                          precision=output_precision)]
                                      + ['{:{width}.{precision}e}'.format(x,width=output_width,precision=output_precision)
                                         for x in [kinetic_energy, sqrt(momenta), sqrt(gradients), sqrt(noise)]])
                else:
                    run_writer.writerow([global_step, i, acc, loss_eval])

        if (i % print_intervals) == 0:
            print('Accuracy at step %s (%s): %s' % (i, global_step, acc))
            #print('Loss at step %s: %s' % (i, loss_eval))
            #print('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
            #print('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
    print("SAMPLED.")


def setup_output_files(FLAGS, nn, config_map):
    """ Prepares the distinct headers for each output file

    :param FLAGS: FLAGS dictionary with command-line parameters
    :param nn: neural network object for obtaining nodes
    :param config_map: configuration dictionary
    :return: CSV writer objects for run and trajectory
    """
    print("Setting up output files")
    if FLAGS.sampler == "StochasticGradientLangevinDynamics":
        header = ['step', 'epoch', 'accuracy', 'loss', 'scaled_gradient', 'scaled_noise']
    elif FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder", "GeometricLangevinAlgorithm_2ndOrder"]:
        header = ['step', 'epoch', 'accuracy', 'loss', 'total_energy', 'kinetic_energy', 'scaled_momentum', 'scaled_gradient', 'scaled_noise']
    else:
        header = ['step', 'epoch', 'accuracy', 'loss']
    run_writer = setup_run_file(FLAGS.run_file, header, config_map)
    trajectory_writer = setup_trajectory_file(FLAGS.trajectory_file,
                                              nn.get("weights").get_shape()[0], nn.get("biases").get_shape()[0],
                                              config_map)
    return run_writer, trajectory_writer

