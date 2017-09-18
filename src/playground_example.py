#!/usr/bion/pyton3
#
# This a command-line version of some of the features found under
# http://playground.tensorflow.org/
# aimed at comparing steepest descent optimizer with a bayesian
# sampling approach.
#
# (C) Frederik Heber 2017-09-18

import argparse, os, sys
import random as rand
import tensorflow as tf
import numpy as np

FLAGS = None

TWOCIRCLES=1
SQUARES=2
TWOCLUSTERS=3
SPIRAL=4

def generate_input_data(dimension, data_type=SPIRAL):
    '''
    Generates the spiral input data where
    data_type decides which type to generate.
    All data resides in the domain [-6,6]^2.
    '''
    returndata = []
    labels = []
    if data_type == TWOCIRCLES:
        for i in range(dimension):
            xcoord = rand.random()*12-6
            ycoord = rand.random()*12-6
            norm = xcoord*xcoord+ycoord*ycoord
            returndata.append([xcoord, ycoord])
            labels.append(int(norm > 9))
    elif data_type == SQUARES:
        print("This is not implemented yet.")
        pass
    elif data_type == TWOCLUSTERS:
        print("This is not implemented yet.")
        pass
    elif data_type == SPIRAL:
        print("This is not implemented yet.")
        pass
    else:
        print("Unknown input data type desired.")
    return [np.array(returndata), np.array(labels)]

def construct_neural_net(input_data, input_labels, steps):
    '''
    Constructs the neural network
    '''
    feature_columns = [tf.feature_column.numeric_column("x", shape=[2])]
    estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
    input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": input_data}, input_labels, batch_size=10, num_epochs=None, shuffle=True)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {"x": input_data}, input_labels, batch_size=10, num_epochs=steps, shuffle=True)
    print("Training network")
    estimator.train(input_fn=input_fn, steps = steps)
    train_metrics = estimator.evaluate(input_fn=train_input_fn)
    print("train metrics: %r"% train_metrics)
        
def main(_):
    print("Generating input data")
    [input_data, input_labels] = generate_input_data(dimension=FLAGS.dimension, data_type=TWOCIRCLES)
    print("Constructing neural network")
    construct_neural_net(input_data, input_labels, FLAGS.max_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, default=10,
        help='Number of steps to run trainer.')
    parser.add_argument('--max_steps', type=int, default=1000,
        help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
        help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
        help='Keep probability for training dropout.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
             'tensorflow/mnist/input_data'),
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
             'tensorflow/mnist/logs/mnist_with_summaries'),
    help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

