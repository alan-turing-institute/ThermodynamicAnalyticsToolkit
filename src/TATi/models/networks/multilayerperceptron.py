#
#    ThermodynamicAnalyticsToolkit - analyze loss manifolds of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#    The TATi authors, see file AUTHORS, have asserted their moral rights.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###

import logging
import tensorflow as tf

from TATi.models.basetype import dds_basetype


class MultiLayerPerceptron(object):
    """ This class adds nodes to the graph to create a multi-layer perceptron.

    """
    @staticmethod
    def weight_variable(shape, seed=None):
        """Create a weight variable, uniform randomly initialized in [-0.5, 0.5].

        :param shape: shape of the weight tensor to create
        :param seed: seed used for initializing the weights
        """
        initial = tf.random_uniform(shape, minval=-0.5, maxval=0.5, seed=seed, dtype=dds_basetype)
        return tf.Variable(initial, dtype=dds_basetype)

    @staticmethod
    def bias_variable(shape):
        """Create a bias variable with appropriate initialization.

        :param shape: shape of the weight tensor to create
        """
        initial = tf.constant(0.1, shape=shape, dtype=dds_basetype)
        return tf.Variable(initial, dtype=dds_basetype)

    @staticmethod
    def _nn_layer(input_tensor, input_dim, output_dim,
                  layer_name, act=tf.nn.relu,
                  trainables_collection=None,
                  seed=None):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.

        :param input_tensor: reference to input layer for this layer
        :param input_dim: number of nodes in `input_layer`
        :param output_dim: number of nodes in the created layer
        :param layer_name: created layer's name
        :param act: activation function to use for the nodes in the created layer
        :param trainables_collection: specific collection to gather all weights of this layer
        :param seed: random number seed for initializing weights
        :return: reference to the created layer
        """
        scope_name = tf.get_default_graph().get_name_scope()
        logging.info("Creating nn layer %s in scope %s with %d, %d"
                     % (layer_name, scope_name, input_dim, output_dim))
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = MultiLayerPerceptron.weight_variable([input_dim, output_dim], seed)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
                if trainables_collection is not None:
                    tf.add_to_collection(trainables_collection, weights)
            with tf.name_scope('biases'):
                biases = MultiLayerPerceptron.bias_variable([output_dim])
                tf.add_to_collection(tf.GraphKeys.BIASES, biases)
                if trainables_collection is not None:
                    tf.add_to_collection(trainables_collection, biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            if act is None:
                return preactivate
            else:
                activations = act(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations

    @staticmethod
    def add_hidden_layers(input_layer, input_dimension, layer_dimensions, keep_prob=None,
                          activation=tf.nn.relu, trainables_collection=None, seed=None):
        """ Add fully connected hidden layers each with an additional dropout layer
         (makes the network robust against overfitting).

        The additional dropped layer will randomly drop samples according to the
         keep probability, i.e. 1 means all samples are keppt, 0 means all samples
         are dropped.

        :param input_layer: reference to the input layer
        :param input_dimension: number of nodes in `input_layer`
        :param keep_prob: reference to the placeholder with the *keep probability* for the dropped layer
        :param layer_dimensions: list of ints giving the number of nodes of each hidden layer,
                entries with dim 0 are skipped entirely
        :param activation: activation function of the hidden layers
        :param trainables_collection: specific collection to gather all weights of this layer
        :param seed: random number see to use for weights (seed is increased by one per layer)
        :return: reference to the last layer created
        """
        current_seed = seed
        last_layer = input_layer
        out_dimension = input_dimension
        for i in range(len(layer_dimensions)):
            if layer_dimensions[i] == 0:
                continue
            if seed is not None:
                current_seed = seed+i
            number = str(i + 1)
            in_dimension = out_dimension
            out_dimension = layer_dimensions[i]
            layer_name = "layer" + number
            current_hidden = MultiLayerPerceptron._nn_layer(
                last_layer, in_dimension, out_dimension, layer_name,
                trainables_collection=trainables_collection, seed=current_seed, act=activation)
            logging.debug(layer_name + " is " + str(current_hidden.get_shape()))

            if keep_prob is not None:
                with tf.name_scope('dropout'):
                    last_layer = tf.nn.dropout(current_hidden, keep_prob)
            else:
                last_layer = current_hidden
            logging.debug("dropped" + number + " is " + str(last_layer.get_shape()))

        return last_layer

    @staticmethod
    def add_output_layer(current_hidden_layer, hidden_out_dimension,
                         output_dimension, activation, trainables_collection=None,
                         seed=None):
        """ Add the output layer giving the predicted values.

        :param current_hidden_layer: last hidden layer which is to be connected to the output layer
        :param hidden_out_dimension: number of nodes in `current_hidden_layer`
        :param output_dimension: number of output nodes
        :param activation: activation function
        :param trainables_collection: specific collection to gather all weights of this layer
        :param seed: random number see to use for weights
        :return: reference to the output layer
        """
        y = MultiLayerPerceptron._nn_layer(
            current_hidden_layer, hidden_out_dimension, output_dimension, 'output',
            trainables_collection=trainables_collection, seed=seed, act=activation)
        logging.debug("y is " + str(y.get_shape()))
        return y

    @staticmethod
    def create(input_layer,
               layer_dimensions, output_dimension,
               trainables_collection=None,
               seed=None,
               keep_prob=None,
               hidden_activation=tf.nn.relu,
               output_activation=tf.nn.tanh):
        """ Creates the neural network model according to the specifications.

        The `input_layer` needs to be given along with its input_dimension.
        The output_layer needs to be specified here as the summaries and
        loss functions depend on them.

        :param input_layer: the input_layer to connect this MLP to
        :param layer_dimensions: a list of ints giving the number of nodes for
            each hidden layer.
        :param output_dimension: the number of nodes in the output layer
        :param trainables_collection: specific collection to gather all weights of this layer
        :param seed: seed for reproducible random values
        :param keep_prob: ref to placeholder for keep probability or None
        :param hidden_activation: activation function for the hidden layer
        :param output_activation: activation function for the output layer
        :return: output layer
        """
        # Mind to only set op-level seeds! As global seed setting depends on the
        # numbering of the nodes in the computational graph which changes when
        # new nodes are added.
        output_seed = None
        if seed is not None:
            tf.set_random_seed(seed)
            output_seed = seed+len(layer_dimensions)

        input_dimension = int(input_layer.get_shape()[-1])
        if layer_dimensions is not None and len(layer_dimensions) != 0:
            last_hidden_layer = \
                MultiLayerPerceptron.add_hidden_layers(
                    input_layer, input_dimension,
                    layer_dimensions, keep_prob, hidden_activation,
                    trainables_collection=trainables_collection, seed=seed)
            # cannot use layer_dimensions[-1] as last entry may be a zero which is skipped
            y = MultiLayerPerceptron.add_output_layer(
                last_hidden_layer, int(last_hidden_layer.get_shape()[-1]),
                output_dimension, output_activation,
                trainables_collection=trainables_collection, seed=output_seed)
        else:
            y = MultiLayerPerceptron.add_output_layer(
                input_layer, input_dimension,
                output_dimension, output_activation,
                trainables_collection=trainables_collection, seed=output_seed)
        return y
