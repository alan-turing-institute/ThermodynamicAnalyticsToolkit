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
from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf
from TATi.samplers.dynamics.baoabsampler import BAOABSampler
from TATi.samplers.dynamics.geometriclangevinalgorithmfirstordersampler import GeometricLangevinAlgorithmFirstOrderSampler
from TATi.samplers.dynamics.geometriclangevinalgorithmsecondordersampler import GeometricLangevinAlgorithmSecondOrderSampler
from TATi.samplers.dynamics.gradientdescent import GradientDescent
from TATi.samplers.dynamics.hamiltonianmontecarlosamplerfirstordersampler import HamiltonianMonteCarloSamplerFirstOrderSampler
from TATi.samplers.dynamics.hamiltonianmontecarlosamplersecondordersampler import HamiltonianMonteCarloSamplerSecondOrderSampler
from TATi.samplers.dynamics.stochasticgradientlangevindynamicssampler import StochasticGradientLangevinDynamicsSampler

from TATi.models.basetype import dds_basetype
from TATi.models.networks.multilayerperceptron import MultiLayerPerceptron
from TATi.samplers.dynamics.covariancecontrolledadaptivelangevinthermostatsampler \
    import CovarianceControlledAdaptiveLangevinThermostat as CCAdLSampler


class NeuralNetwork(object):
    """This class encapsulates the construction of the neural network.
    
    Note that functions both for creating the input and the output layer are
     not contained. These depend on the specifics of the dataset for which
     the neural network is to be trained and hence belong there.
    
    TensorFlow obtains its name by the flow of tensors through the graph
     of a neural network from input to output. A tensor can be a number, a
     vector, a matrix or higher modes.
     The size of training set, i.e. the batch size, is always another dimension.
     In essence, if you train with a single input node, having ten labeled
     items, then you have a [10, 1] tensor, i.e. two-dimensional with 10
      components in the first dimension and 1 component in the second.
    
    Internally, to actually perform computations with this neural network
     TensorFlow then uses as so-called `computational graph`. A node in
     this graph represents a specific operation that depends on certain input
     variables and generates output. The input variables have to be represented
     by nodes as well.
     This *dependence* graph allows the TensorFlow engine to exactly determine
     which nodes it needs to evaluate for a certain operation. Moreover,
     evaluations can even be done in parallel to a certain extent.
    
    Nodes can be constants, variables or placeholders, i.e. values fed in by
     the user. Moreover, nodes can be the results of operations such as
     multiplication, summation, and so on.
     Naturally, each input and output node is a node in the `computational
     graph`. However, even the training method itself is also a set of nodes
     that depend on other nodes, such as the weight of each layer, the input
     and output layer and the true labels.

    Args:

    Returns:

    """
    def __init__(self):
        self.placeholder_nodes = {}
        """Lookup dictionary for input nodes, in TensorFlow parlance called placeholders."""
        self.summary_nodes = {}
        """Lookup dictionary for summary nodes, specific to TensorFlow. """
        self.loss_nodes = {}
        """ Lookup dictionary for the loss nodes, to tell TensorFlow which to train on"""

    def get(self, keyname):
        """Retrieve a node by name from the TensorFlow computational graph.

        Args:
          keyname: name of node to retrieve

        Returns:
          node if found or None

        """
        if keyname in self.summary_nodes:
            return self.summary_nodes[keyname]
        elif keyname in self.placeholder_nodes:
            return self.placeholder_nodes[keyname]
        else:
            logging.warning(" Could not find node "+keyname+" in dictionary.")
            return None

    def get_list_of_nodes(self, keys):
        """This returns for a list of node names the list of nodes.
        
        We assert that none of the Nodes is None.

        Args:
          keys: list with node keys, i.e. names

        Returns:
          list of nodes

        """
        test_nodes = list(map(lambda key: self.get(key), keys))
        for key, node in zip(keys, test_nodes):
            if node is None:
                raise AssertionError("Node " + key + " could not be retrieved from dict or is None.")
        return test_nodes

    def get_dict_of_nodes(self, keys):
        """Returns a dict with access to nodes by name.

        Args:
          keys: names of the nodes

        Returns:
          dictionary

        """
        return dict(zip(keys, self.get_list_of_nodes(keys)))

    def create(self, input_layer,
               layer_dimensions, output_dimension, labels,
               trainables_collection=None,
               seed=None,
               keep_prob=None,
               hidden_activation=tf.nn.relu,
               output_activation=tf.nn.tanh,
               loss_name="mean_squared"):
        """Creates the neural network model according to the specifications.
        
        The `input_layer` needs to be given along with its input_dimension.
        The output_layer needs to be specified here as the summaries and
        loss functions depend on them.

        Args:
          input_layer: the input_layer
          layer_dimensions: a list of ints giving the number of nodes for
        each hidden layer.
          output_dimension: the number of nodes in the output layer
          labels: node to labels for calculating loss and accuracy
          trainables_collection: specific collection to gather all weights of this layer (Default value = None)
          seed: seed for reproducible random values (Default value = None)
          keep_prob: ref to placeholder for keep probability or None (Default value = None)
          hidden_activation: activation function for the hidden layer (Default value = tf.nn.relu)
          output_activation: activation function for the output layer (Default value = tf.nn.tanh)
          loss_name: name of loss to use in training, see :method:`NeuralNetwork.add_losses` (Default value = "mean_squared")

        Returns:

        """
        self.summary_nodes.clear()

        y = MultiLayerPerceptron.create(input_layer, layer_dimensions, output_dimension,
                                        trainables_collection=trainables_collection,
                                        seed=seed, keep_prob=keep_prob,
                                        hidden_activation=hidden_activation,
                                        output_activation=output_activation)

        self.placeholder_nodes["y"] = y
        self.summary_nodes['y'] = y

        self.add_losses(y, labels)
        loss = self.set_loss_function(loss_name)

        return loss

    @staticmethod
    def add_true_labels(output_dimension):
        """Adds the known labels as placeholder nodes to the graph.

        Args:
          output_dimension: number of output nodes

        Returns:
          reference to created output layer

        """
        y_ = tf.placeholder(dds_basetype, [None, output_dimension], name='y-input')
        logging.debug("y_ is "+str(y_.get_shape()))
        return y_

    @staticmethod
    def add_accuracy_summary(y, y_, output_type = 0):
        """Add nodes to the graph to calculate the accuracy for the dataset.
        
        The accuracy is the difference between the predicted label and the true
        label as mean average, i.e. 0.5 is random, 1 is the best, and 0 means
        you have the labels wrong way round :)
        
        Note that the accuracy node can be obtained via :method:`neuralnetwork.get`.
        For evaluation it needs to be given to a tensorflow.Session.run() which
        will return the evaluated node given a dataset.

        Args:
          y: predicted labels
          y_: true labels
          output_type: type of label set (i.e. {-1,1} or {0,1}^c) (Default value = 0)

        Returns:
          accuracy node

        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                if output_type == "binary_classification":
                    correct_prediction = tf.equal(tf.sign(y), tf.sign(y_))
                elif output_type == "onehot_multi_classification":
                    correct_prediction = tf.equal(tf.argmax(y,1),
                                                  tf.argmax(y_,1))
                else:
                    logging.error("Unknown output type.")
                    assert(0)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, dds_basetype))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def _prepare_global_step(self):
        """Adds the global_step node to the graph.

        Args:

        Returns:
          global_step node

        """
        # have this outside scope as it is used by both training and learning
        if 'global_step' not in self.summary_nodes.keys():
            global_step = tf.Variable(0, trainable=False, name="global_step")
            tf.add_to_collection("Variables_to_Save", global_step)
            self.summary_nodes['global_step'] = global_step
        else:
            global_step = self.summary_nodes['global_step']
        return global_step

    def add_sample_method(self, loss, sampling_method, seed,
                          prior, sigma=None, sigmaA=None):
        """Prepares adding nodes for training the neural network.

        Args:
          loss: node for the desired loss function to minimize during training
          sampling_method: name of the sampler method, e.g. GradientDescent
          seed: seed value for the random number generator to obtain reproducible runs
          prior: dict with keys factor, lower_boundary and upper_boundary that
        specifies a wall-repelling force to ensure a prior on the parameters
          sigma: scale of noise injected to momentum per step for CCaDL only (Default value = None)
          sigmaA: scale of noise in convex combination for CCaDL only (Default value = None)

        Returns:

        """
        global_step = self._prepare_global_step()
        sampler = self._prepare_sampler(loss, sampling_method,
                                        seed, prior, sigma, sigmaA)
        self._finalize_sample_method(loss, sampler, global_step)

    def _prepare_sampler(self, loss, sampling_method, seed,
                         prior, sigma=None, sigmaA=None):
        """Prepares the sampler instance, adding also all placeholder nodes it requires.

        Args:
          loss: node for the desired loss function to minimize during training
          sampling_method: name of the sampler method, e.g. GradientDescent
          seed: seed value for the random number generator to obtain reproducible runs
          prior: dict with keys factor, lower_boundary and upper_boundary that
        specifies a wall-repelling force to ensure a prior on the parameters
          sigma: scale of noise injected to momentum per step for CCaDL only (Default value = None)
          sigmaA: scale of noise in convex combination for CCaDL only (Default value = None)

        Returns:
          created sampler instance

        """
        with tf.name_scope('sample'):
            # DON'T add placeholders only sometimes, e.g. when only a specific sampler
            # requires it. Always add them and only sometimes use them!
            step_width = tf.placeholder(dds_basetype, name="step_width")
            tf.summary.scalar('step_width', step_width)
            self.placeholder_nodes['step_width'] = step_width

            next_eval_step = tf.placeholder(tf.int64, name="next_eval_step")
            tf.summary.scalar('next_eval_step', next_eval_step)
            self.placeholder_nodes['next_eval_step'] = next_eval_step

            hd_steps = tf.placeholder(tf.int64, name="hamiltonian_dynamics_steps")
            tf.summary.scalar('hamiltonian_dynamics_steps', hd_steps)
            self.placeholder_nodes['hamiltonian_dynamics_steps'] = hd_steps

            current_step = tf.placeholder(tf.int64, name="current_step")
            tf.summary.scalar('current_step', current_step)
            self.placeholder_nodes['current_step'] = current_step

            inverse_temperature = tf.placeholder(dds_basetype, name="inverse_temperature")
            tf.summary.scalar('inverse_temperature', inverse_temperature)
            self.placeholder_nodes['inverse_temperature'] = inverse_temperature

            friction_constant = tf.placeholder(dds_basetype, name="friction_constant")
            tf.summary.scalar('friction_constant', friction_constant)
            self.placeholder_nodes['friction_constant'] = friction_constant

            covariance_blending = tf.placeholder(dds_basetype, name="covariance_blending")
            tf.summary.scalar('covariance_blending', covariance_blending)
            self.placeholder_nodes['covariance_blending'] = covariance_blending

            if sampling_method == "StochasticGradientLangevinDynamics":
                sampler = StochasticGradientLangevinDynamicsSampler(covariance_blending,
                                                                    step_width, inverse_temperature,
                                                                    seed=seed)
            elif sampling_method == "GeometricLangevinAlgorithm_1stOrder":
                sampler = GeometricLangevinAlgorithmFirstOrderSampler(covariance_blending,
                                                                      step_width, inverse_temperature, friction_constant,
                                                                      seed=seed)
            elif sampling_method == "GeometricLangevinAlgorithm_2ndOrder":
                sampler = GeometricLangevinAlgorithmSecondOrderSampler(covariance_blending,
                                                                       step_width, inverse_temperature, friction_constant,
                                                                       seed=seed)
            elif "HamiltonianMonteCarlo" in sampling_method:
                if seed is not None:
                    np.random.seed(seed)
                accept_seed = int(np.random.uniform(low=0,high=67108864))
                if sampling_method  == "HamiltonianMonteCarlo_1stOrder":
                    sampler = HamiltonianMonteCarloSamplerFirstOrderSampler(
                        covariance_blending, step_width, inverse_temperature, loss,
                        current_step, next_eval_step, accept_seed=accept_seed, seed=seed)
                elif sampling_method == "HamiltonianMonteCarlo_2ndOrder":
                    sampler = HamiltonianMonteCarloSamplerSecondOrderSampler(
                        covariance_blending, step_width, inverse_temperature,
                        loss, current_step, next_eval_step, hd_steps,
                        accept_seed=accept_seed, seed=seed)
                else:
                    raise NotImplementedError("The HMC sampler %s is unknown" % (sampling_method))
            elif sampling_method == "BAOAB":
                sampler = BAOABSampler(covariance_blending,
                                       step_width, inverse_temperature, friction_constant,
                                       seed=seed)
            elif sampling_method == "CovarianceControlledAdaptiveLangevinThermostat":
                sampler = CCAdLSampler(covariance_blending,
                                       step_width, inverse_temperature, friction_constant,
                                       sigma=sigma, sigmaA=sigmaA, seed=seed)
            else:
                raise NotImplementedError("Unknown sampler")
            if len(prior) != 0:
                sampler.set_prior(prior)

            return sampler

    def _finalize_sample_method(self, loss, sampler, global_step):
        """Adds nodes for training the neural network.

        Args:
          loss: node for the desired loss function to minimize during training
          sampler: sampler instance to use for sampling
          global_step: global_step node

        Returns:

        """
        with tf.name_scope('sample'):
            trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            grads_and_vars = sampler.compute_and_check_gradients(loss, var_list=trainables)
            train_step = sampler.apply_gradients(grads_and_vars, global_step=global_step,
                                                 name=sampler.get_name())

            # DON'T put the nodes in there before the minimize call!
            # only after minimize was .._apply_dense() called and the nodes are ready
            self.summary_nodes['sample_step'] = train_step

    def add_train_method(self, loss, optimizer_method,
                          prior):
        """Adds nodes for training the neural network using an optimizer.

        Args:
          loss: node for the desired loss function to minimize during training
          optimizer_method: name of the optimizer method, e.g. GradientDescent
          prior: dict with keys factor, lower_boundary and upper_boundary that
        specifies a wall-repelling force to ensure a prior on the parameters

        Returns:

        """
        global_step = self._prepare_global_step()
        optimizer = self._prepare_optimizer(loss, optimizer_method, prior)
        self._finalize_train_method(loss, optimizer, global_step)

    def _prepare_optimizer(self, loss, optimizer_method,
                          prior):
        """Prepares optimizer instances, adding the placeholder it needs.

        Args:
          loss: node for the desired loss function to minimize during training
          optimizer_method: name of the optimizer method, e.g. GradientDescent
          prior: dict with keys factor, lower_boundary and upper_boundary that
        specifies a wall-repelling force to ensure a prior on the parameters

        Returns:
          created optimizer instance

        """
        # have this outside scope as it is used by both training and learning
        with tf.name_scope('train'):
            # DON'T add placeholders only sometimes, e.g. when only a specific optimizer
            # requires it. Always add them and only sometimes use them!
            learning_rate = tf.placeholder(dds_basetype, name="learning_rate")
            tf.summary.scalar('learning_rate', learning_rate)
            self.placeholder_nodes['learning_rate'] = learning_rate

            if optimizer_method == "GradientDescent":
                optimizer = GradientDescent(learning_rate)
            else:
                raise NotImplementedError("Unknown optimizer_method")
            if len(prior) != 0:
                optimizer.set_prior(prior)

            return optimizer

    def _finalize_train_method(self, loss, optimizer, global_step):
        """Prepares nodes for training the neural network using an optimizer.

        Args:
          loss: node for the desired loss function to minimize during training
          optimizer: optimizer instance
          global_step: global_step node

        Returns:

        """
        with tf.name_scope('train'):
            trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            train_step = optimizer.minimize(loss, global_step=global_step, var_list=trainables)

            # DON'T put the nodes in there before the minimize call!
            # only after minimize was .._apply_dense() called and the nodes are ready
            self.summary_nodes['train_step'] = train_step

    def set_loss_function(self, loss_name):
        """Set the loss function to minimize when optimizing.
        
        Note that the loss node can be obtained via :method:`neuralnetwork.get`.
        For evaluation it needs to be given to a tensorflow.Session.run() which
        will return the evaluated node given a dataset.

        Args:
          loss_name: name of the loss function

        Returns:
          loss node for :method:`tensorflow.train`

        """
        if loss_name not in self.loss_nodes:
            raise NotImplementedError
        with tf.name_scope('total'):
            loss = self.loss_nodes[loss_name]
            tf.summary.scalar('loss', loss)
            self.summary_nodes["loss"] = loss
        return loss

    def add_losses(self, y, y_):
        """Add nodes to the graph to calculate losses for the dataset.

        Args:
          y: predicted labels
          y_: true labels

        Returns:

        """
        with tf.name_scope('loss'):
            sigmoid_cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y))
            absolute_difference = tf.losses.absolute_difference(labels=y_, predictions=y)
            if LooseVersion(tf.__version__) < LooseVersion("1.5.0"):
                cosine_distance = tf.losses.cosine_distance(labels=y_, predictions=y, dim=1)
                softmax_cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
            else:
                cosine_distance = tf.losses.cosine_distance(labels=y_, predictions=y, axis=1)
                softmax_cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
            hinge_loss = tf.losses.hinge_loss(labels=y_, logits=y)
            log_loss = tf.losses.log_loss(labels=y_, predictions=y)
            mean_squared = tf.losses.mean_squared_error(labels=y_, predictions=y)
        tf.summary.scalar('softmax_cross_entropy', softmax_cross_entropy)

        self.loss_nodes["mean_squared"] = mean_squared
        self.loss_nodes["log_loss"] = log_loss
        self.loss_nodes["hinge_loss"] = hinge_loss
        self.loss_nodes["cosine_distance"] = cosine_distance
        self.loss_nodes["absolute_difference"] = absolute_difference
        self.loss_nodes["sigmoid_cross_entropy"] = sigmoid_cross_entropy
        self.loss_nodes["softmax_cross_entropy"] = softmax_cross_entropy

    def add_keep_probability(self):
        """Adds a placeholder node for the keep probability of dropped layers.
        
        See :method:`neuralnetwork.add_hidden_layers`

        Args:

        Returns:
          reference to created node

        """
        keep_prob = tf.placeholder(dds_basetype, name="keep_probability")
        with tf.name_scope('dropout'):
            tf.summary.scalar('dropout_keep_probability', keep_prob)
        self.placeholder_nodes['keep_probability'] = keep_prob
        return keep_prob

    def add_writers(self, sess, log_dir):
        """Adds log writers.
        
        Logs allow to visualize and debug the computational graph using
        TensorBoard (part of the Tensorflow package). Logs are files written
        to disk that contain all summary information.

        Args:
          sess: Tensorflow Session
          log_dir: string giving directory to write files to

        Returns:

        """
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        self.summary_nodes["train_writer"] = train_writer
        test_writer = tf.summary.FileWriter(log_dir + '/test')
        self.summary_nodes["test_writer"] = test_writer
        
    @staticmethod
    def init_graph(sess):
        """Initializes global variables in the computational graph.

        Args:
          sess: Tensorflow Session

        Returns:

        """
        logging.debug ("Initializing global variables")
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()]
        )

    @staticmethod
    def variable_summaries(var):
        """Attach a lot of summaries (mean, stddev, min, max) to a given tensor
        for TensorBoard visualization.

        Args:
          var: ref to the tensor variable to summarize

        Returns:

        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    @staticmethod
    def get_activations():
        """Returns a dictionary with all known activation functions

        Args:

        Returns:
          dictionary with activations

        """
        activations = {
            "tanh": tf.nn.tanh,
            "sigmoid": tf.nn.sigmoid,
            "softplus": tf.nn.softplus,
            "softsign": tf.nn.softsign,
            "elu": tf.nn.elu,
            "relu6": tf.nn.relu6,
            "relu": tf.nn.relu,
            "linear": tf.identity
        }
        return activations


