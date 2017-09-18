#!/usr/bion/pyton3
#
# This a command-line version of some of the features found under
# http://playground.tensorflow.org/
# aimed at comparing steepest descent optimizer with a bayesian
# sampling approach.
#
# (C) Frederik Heber 2017-09-18

import tensorflow as tf

def construct_neural_net():
	'''
	Constructs the neural network
	'''
	pass


TWOCIRCLES=1
SQUARES=2
TWOCLUSTERS=3
SPIRAL=4

def generate_input_data(dimension, data_type=SPRIAL):
	'''
	Generates the spiral input data where
	data_type decides which type to generate.
	All data resides in the domain [-6,6]^2.
	'''
	y_ = tf.placeholder(tf.float32, [None, dimension])
	if data_type == TWOCIRCLES:
		pass
	return y_


def main(_):
	y = generate_input_data()
	nn.construct_neural_net()
	# TODO: optimizer? ...
	nn.train()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
	                    default=False,
	                    help='If true, uses fake data for unit testing.')
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

