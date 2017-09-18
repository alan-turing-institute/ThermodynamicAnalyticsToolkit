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
import math
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

FLAGS = None

TWOCIRCLES=0
SQUARES=1
TWOCLUSTERS=2
SPIRAL=3

def generate_input_data(dimension, noise, data_type=SPIRAL):
    '''
    Generates the spiral input data where
    data_type decides which type to generate.
    All data resides in the domain [-6,6]^2.
    '''
    returndata = []
    labels = []
    r = 5
    if data_type == TWOCIRCLES:
        for label in [1,-1]:
            for i in range(int(dimension/2)):
                if label == 1:
                    radius = np.random.uniform(0,r*0.5)
                else:
                    radius = np.random.uniform(r*0.7, r)
                angle = np.random.uniform(0,2*math.pi)
                coords = [radius * math.sin(angle), radius * math.cos(angle)]
                noisecoords = np.random.uniform(-r,r,2)*noise
                norm = (coords[0]+noisecoords[0])*(coords[0]+noisecoords[0])+(coords[1]+noisecoords[1])*(coords[1]+noisecoords[1])
                returndata.append(coords)
                labels.append(1 if (norm < r*r*.25) else -1)
                #print(str(returndata[-1])+" with norm "+str(norm)+" and radius "+str(radius)+": "+str(labels[-1]))
    elif data_type == SQUARES:
        for i in range(dimension):
            coords = np.random.uniform(-r,r,2)
            padding = .3
            coords[0] += padding * (1 if (coords[0] > 0) else -1)
            coords[1] += padding * (1 if (coords[1] > 0) else -1)
            noisecoords = np.random.uniform(-r,r,2)*noise
            returndata.append(coords)
            labels.append(1 if ((coords[0]+noisecoords[0])*(coords[1]+noisecoords[1]) >= 0) else -1)
    elif data_type == TWOCLUSTERS:
        variance = 0.5+noise*(3.5*2)
        for label in [1,-1]:
            for i in range(int(dimension/2)):
                coords = np.random.normal(label*2,variance,2)
                returndata.append(coords)
                labels.append(label)
    elif data_type == SPIRAL:
        for deltaT in [0, math.pi]:
            for i in range(int(dimension/2)):
                radius = i/dimension*r
                t = 3.5 * i/dimension* 2*math.pi + deltaT
                coords = [radius*math.sin(t)+np.random.uniform(-1,1)*noise,
                          radius*math.cos(t)+np.random.uniform(-1,1)*noise]
                returndata.append(coords)
                labels.append(1 if (deltaT == 0) else -1)
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
    [input_data, input_labels] = generate_input_data(
        dimension=FLAGS.dimension,
        noise=FLAGS.noise,
        data_type=FLAGS.data_type)
    print("Displaying data")
    plt.scatter([val[0] for val in input_data], [val[1] for val in input_data],
                s=FLAGS.dimension,
                c=[('r' if (label == 1) else 'b') for label in input_labels])
    plt.show()
    #print("Constructing neural network")
    #construct_neural_net(input_data, input_labels, FLAGS.max_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, default=10,
        help='Number of samples to generate.')
    parser.add_argument('--max_steps', type=int, default=1000,
        help='Number of steps to run trainer.')
    parser.add_argument('--noise', type=float, default=0,
        help='Amount of noise in [0,1] to use.')
    parser.add_argument('--data_type', type=int, default=SPIRAL,
        help='Which data set to use: two circles, squares, two clusters, spiral.')
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

