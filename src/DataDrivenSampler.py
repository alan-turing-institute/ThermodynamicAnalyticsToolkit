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


def parse_parameters():
    """ Sets up the argument parser for parsing command line parameters into dictionary

    :return: dictionary with parameter names as keys, unrecognized parameters
    """
    parser = argparse.ArgumentParser()
    # please adhere to alphabetical ordering
    parser.add_argument('--csv_file', type=str, default=None,
        help='CSV file name to output accuracy and loss values.')
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
    parser.add_argument('--sampler', type=str, default="StochasticGradientLangevinDynamics",
        help='Choose the sampler to use for sampling: StochasticGradientLangevinDynamics')
    parser.add_argument('--seed', type=int, default=None,
        help='Seed to use for random number generators.')
    parser.add_argument('--step_width', type=float, default=0.03,
        help='step width \Delta t to use, e.g. 0.01')
    parser.add_argument('--trajectory_file', type=str, default=None,
        help='CSV file name to output trajectories of sampling, i.e. weights and evaluated loss function.')
    parser.add_argument('--version', action="store_true",
        help='Gives version information')
    return parser.parse_known_args()


def sample(FLAGS, ds, sess, nn, xinput, csv_writer, trajectory_writer, config_map):
    """ Performs the actual sampling of the neural network `nn` given a dataset `ds` and a
    Session `session`.

    :param FLAGS: FLAGS dictionary with command-line parameters
    :param ds: dataset
    :param sess: Session object
    :param nn: neural network
    :param xinput: input nodes of neural network
    :param csv_writer: run csv writer
    :param trajectory_writer: trajectory csv writer
    :param config_map: configuration dictionary
    """
    placeholder_nodes = nn.get_dict_of_nodes(
        ["friction_constant", "inverse_temperature", "step_width", "y_"])
    test_nodes = nn.get_list_of_nodes(["merged", "train_step", "accuracy", "global_step", "loss", "y_", "y"])
    if FLAGS.sampler == "StochasticGradientLangevinDynamics":
        noise_nodes = nn.get_list_of_nodes(["scaled_gradient", "scaled_noise"])
    if FLAGS.sampler == "StochasticMomentumLangevin":
        mom_noise_nodes = nn.get_list_of_nodes(["kinetic_energy", "scaled_momentum", "scaled_gradient", "scaled_noise"])

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
    for i in range(FLAGS.max_steps):
        print("Current step is "+str(i))
        test_xs, test_ys = ds.get_testset()
        feed_dict={
            xinput: test_xs, placeholder_nodes["y_"]: test_ys,
            placeholder_nodes["step_width"]: FLAGS.step_width,
            placeholder_nodes["inverse_temperature"]: FLAGS.inverse_temperature,
            placeholder_nodes["friction_constant"]: FLAGS.friction_constant
        }
        # print("Testset is x: "+str(test_xs[0:5])+", y: "+str(test_ys[0:5]))
        summary, _, acc, global_step, loss_eval, y_true_eval, y_eval = sess.run(test_nodes, feed_dict=feed_dict)
        if i % FLAGS.every_nth == 0:
            if config_map["do_write_trajectory_file"]:
                weights_eval, biases_eval = sess.run(
                    [nn.get("weights"), nn.get("biases")],
                    feed_dict=feed_dict)
                trajectory_writer.writerow(
                    [global_step, loss_eval]
                    + [item for sublist in weights_eval for item in sublist]
                    + [item for item in biases_eval])

            if config_map["do_write_csv_file"]:
                if FLAGS.sampler == "StochasticGradientLangevinDynamics":
                    csv_writer.writerow([global_step, i, acc, loss_eval]
                                        + sess.run(noise_nodes,feed_dict=feed_dict))
                elif FLAGS.sampler == "StochasticMomentumLangevin":
                  kinetic_energy, scaled_mom, scaled_grad, scaled_noise = \
                      sess.run(mom_noise_nodes, feed_dict=feed_dict)
                  csv_writer.writerow([global_step, i, acc, loss_eval]
                                      + [loss_eval+kinetic_energy]
                                      + [kinetic_energy, scaled_mom, scaled_grad, scaled_noise])
                else:
                  csv_writer.writerow([global_step, i, acc, loss_eval])

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
    elif FLAGS.sampler == "StochasticMomentumLangevin":
        header = ['step', 'epoch', 'accuracy', 'loss', 'total_energy', 'kinetic_energy', 'scaled_momentum', 'scaled_gradient', 'scaled_noise']
    else:
        header = ['step', 'epoch', 'accuracy', 'loss']
    csv_writer = setup_run_file(FLAGS.csv_file, header, config_map)
    trajectory_writer = setup_trajectory_file(FLAGS.trajectory_file,
                                              nn.get("weights").get_shape()[0], nn.get("biases").get_shape()[0],
                                              config_map)
    return csv_writer, trajectory_writer
