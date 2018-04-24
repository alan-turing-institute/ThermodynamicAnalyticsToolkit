import logging
import numpy as np
import tensorflow as tf

from distutils.version import LooseVersion

from TATi.models.basetype import dds_basetype
from TATi.samplers.covariancecontrolledadaptivelangevinthermostatsampler \
    import CovarianceControlledAdaptiveLangevinThermostat as CCAdLSampler
from TATi.samplers.baoabsampler import BAOABSampler
from TATi.samplers.geometriclangevinalgorithmfirstordersampler import GeometricLangevinAlgorithmFirstOrderSampler
from TATi.samplers.geometriclangevinalgorithmsecondordersampler import GeometricLangevinAlgorithmSecondOrderSampler
from TATi.samplers.hamiltonianmontecarlosampler import HamiltonianMonteCarloSampler
from TATi.samplers.gradientdescent import GradientDescent
from TATi.samplers.stochasticgradientlangevindynamicssampler import StochasticGradientLangevinDynamicsSampler


class NeuralNetwork(object):
    """ This class encapsulates the construction of the neural network.

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
    """
    def __init__(self):
        self.placeholder_nodes = {}
        """Lookup dictionary for input nodes, in TensorFlow parlance called placeholders."""
        self.summary_nodes = {}
        """Lookup dictionary for summary nodes, specific to TensorFlow. """
        self.loss_nodes = {}
        """ Lookup dictionary for the loss nodes, to tell TensorFlow which to train on"""

    def get(self, keyname):
        """ Retrieve a node by name from the TensorFlow computational graph.

        :param keyname: name of node to retrieve
        :return: node if found or None
        """
        if keyname in self.summary_nodes:
            return self.summary_nodes[keyname]
        elif keyname in self.placeholder_nodes:
            return self.placeholder_nodes[keyname]
        else:
            logging.warning(" Could not find node "+keyname+" in dictionary.")
            return None

    def get_list_of_nodes(self, keys):
        """ This returns for a list of node names the list of nodes.

        We assert that none of the Nodes is None.

        :param keys: list with node keys, i.e. names
        :return: list of nodes
        """
        test_nodes = list(map(lambda key: self.get(key), keys))
        for key, node in zip(keys, test_nodes):
            if node is None:
                logging.info ("Node " + key + " could not be retrieved from dict or is None.")
                raise AssertionError
        return test_nodes

    def get_dict_of_nodes(self, keys):
        """ Returns a dict with access to nodes by name.

        :param keys: names of the nodes
        :return: dictionary
        """
        return dict(zip(keys, self.get_list_of_nodes(keys)))

    def create(self, input_layer,
               layer_dimensions, output_dimension, labels,
               trainables_collection=None,
               seed=None,
               add_dropped_layer=False,
               hidden_activation=tf.nn.relu,
               output_activation=tf.nn.tanh,
               loss_name="mean_squared"):
        """ Creates the neural network model according to the specifications.

        The `input_layer` needs to be given along with its input_dimension.
        The output_layer needs to be specified here as the summaries and
        loss functions depend on them.

        :param input_layer: the input_layer
        :param layer_dimensions: a list of ints giving the number of nodes for
            each hidden layer.
        :param output_dimension: the number of nodes in the output layer
        :param labels: node to labels for calculating loss and accuracy
        :param trainables_collection: specific collection to gather all weights of this layer
        :param seed: seed for reproducible random values
        :param add_dropped_layer: whether to add dropped layer or not to protect against overfitting
        :param hidden_activation: activation function for the hidden layer
        :param output_activation: activation function for the output layer
        :param loss_name: name of loss to use in training, see :method:`NeuralNetwork.add_losses`
        """
        self.summary_nodes.clear()
        # DON'T: this will produce ever the same random number tensor!
        # only set op-level seeds!
        output_seed = None
        if seed is not None:
            tf.set_random_seed(seed)
            output_seed = seed+len(layer_dimensions)

        input_dimension = int(input_layer.get_shape()[-1])
        if add_dropped_layer:
            keep_prob = self.add_keep_probability()
        else:
            keep_prob = None
        if layer_dimensions is not None and len(layer_dimensions) != 0:
            last_hidden_layer = \
                self.add_hidden_layers(input_layer, input_dimension,
                                       layer_dimensions, keep_prob, hidden_activation,
                                       trainables_collection=trainables_collection,
                                       seed=seed)
            y = self.add_output_layer(last_hidden_layer, layer_dimensions[-1],
                                      output_dimension, output_activation,
                                      trainables_collection=trainables_collection,
                                      seed=output_seed)
        else:
            y = self.add_output_layer(input_layer, input_dimension,
                                      output_dimension, output_activation,
                                      trainables_collection=trainables_collection,
                                      seed=output_seed)
        self.placeholder_nodes["y"] = y

        logging.debug ("Creating summaries")
        self.add_losses(y, labels)
        loss = self.set_loss_function(loss_name)
        self.add_accuracy_summary(y, labels)

        return loss

    @staticmethod
    def add_true_labels(output_dimension):
        """ Adds the known labels as placeholder nodes to the graph.

        :param output_dimension: number of output nodes
        :return: reference to created output layer
        """
        y_ = tf.placeholder(dds_basetype, [None, output_dimension], name='y-input')
        logging.debug("y_ is "+str(y_.get_shape()))
        return y_

    def add_accuracy_summary(self, y, y_):
        """ Add nodes to the graph to calculate the accuracy for the dataset.

        The accuracy is the difference between the predicted label and the true
        label as mean average, i.e. 0.5 is random, 1 is the best, and 0 means
        you have the labels wrong way round :)

        Note that the accuracy node can be obtained via :method:`neuralnetwork.get`.
        For evaluation it needs to be given to a tensorflow.Session.run() which
        will return the evaluated node given a dataset.

        :param y: predicted labels
        :param y_: true labels
        """
        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.sign(y), tf.sign(y_))
                self.summary_nodes['correct_prediction'] = correct_prediction
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, dds_basetype))
                self.summary_nodes['accuracy'] = accuracy
        tf.summary.scalar('accuracy', accuracy)

    def _prepare_global_step(self):
        """ Adds the global_step node to the graph.

        :return: global_step node
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
        """ Prepares adding nodes for training the neural network.

        :param loss: node for the desired loss function to minimize during training
        :param sampling_method: name of the sampler method, e.g. GradientDescent
        :param seed: seed value for the random number generator to obtain reproducible runs
        :param prior: dict with keys factor, lower_boundary and upper_boundary that
                specifies a wall-repelling force to ensure a prior on the parameters
        :param sigma: scale of noise injected to momentum per step for CCaDL only
        :param sigmaA: scale of noise in convex combination for CCaDL only
        """
        global_step = self._prepare_global_step()
        sampler = self._prepare_sampler(loss, sampling_method, seed, prior, sigma, sigmaA)
        self._finalize_sample_method(loss, sampler, global_step)

    def _prepare_sampler(self, loss, sampling_method, seed,
                         prior, sigma=None, sigmaA=None,
                         covariance_blending=0.):
        """ Prepares the sampler instance, adding also all placeholder nodes it requires.

        :param loss: node for the desired loss function to minimize during training
        :param sampling_method: name of the sampler method, e.g. GradientDescent
        :param seed: seed value for the random number generator to obtain reproducible runs
        :param prior: dict with keys factor, lower_boundary and upper_boundary that
                specifies a wall-repelling force to ensure a prior on the parameters
        :param sigma: scale of noise injected to momentum per step for CCaDL only
        :param sigmaA: scale of noise in convex combination for CCaDL only
        :param covariance_blending: eta value for blending of identity and covariance
                matrix from other replica
        :return: created sampler instance
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
                                                                    step_width, inverse_temperature, seed=seed)
            elif sampling_method == "GeometricLangevinAlgorithm_1stOrder":
                sampler = GeometricLangevinAlgorithmFirstOrderSampler(covariance_blending,
                                                                      step_width, inverse_temperature, friction_constant, seed=seed)
            elif sampling_method == "GeometricLangevinAlgorithm_2ndOrder":
                sampler = GeometricLangevinAlgorithmSecondOrderSampler(covariance_blending,
                                                                       step_width, inverse_temperature, friction_constant, seed=seed)
            elif sampling_method == "HamiltonianMonteCarlo":
                if seed is not None:
                    np.random.seed(seed)
                accept_seed = np.random.uniform(low=0,high=67108864)
                sampler = HamiltonianMonteCarloSampler(covariance_blending,
                                                       step_width, inverse_temperature, current_step, next_eval_step, accept_seed=accept_seed, seed=seed)
            elif sampling_method == "BAOAB":
                sampler = BAOABSampler(covariance_blending,
                                       step_width, inverse_temperature, friction_constant, seed=seed)
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
        """ Adds nodes for training the neural network.

        :param loss: node for the desired loss function to minimize during training
        :param sampler: sampler instance to use for sampling
        :param global_step: global_step node
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
        """ Adds nodes for training the neural network using an optimizer.

        :param loss: node for the desired loss function to minimize during training
        :param optimizer_method: name of the optimizer method, e.g. GradientDescent
        :param prior: dict with keys factor, lower_boundary and upper_boundary that
                specifies a wall-repelling force to ensure a prior on the parameters
        """
        global_step = self._prepare_global_step()
        optimizer = self._prepare_optimizer(loss, optimizer_method, prior)
        self._finalize_train_method(loss, optimizer, global_step)

    def _prepare_optimizer(self, loss, optimizer_method,
                          prior):
        """ Prepares optimizer instances, adding the placeholder it needs.

        :param loss: node for the desired loss function to minimize during training
        :param optimizer_method: name of the optimizer method, e.g. GradientDescent
        :param prior: dict with keys factor, lower_boundary and upper_boundary that
                specifies a wall-repelling force to ensure a prior on the parameters
        :return: created optimizer instance
        """
        # have this outside scope as it is used by both training and learning
        with tf.name_scope('train'):
            # DON'T add placeholders only sometimes, e.g. when only a specific optimizer
            # requires it. Always add them and only sometimes use them!
            step_width = tf.placeholder(dds_basetype, name="learning_rate")
            tf.summary.scalar('learning_rate', step_width)
            self.placeholder_nodes['learning_rate'] = step_width

            if optimizer_method == "GradientDescent":
                optimizer = GradientDescent(step_width)
            else:
                raise NotImplementedError("Unknown optimizer_method")
            if len(prior) != 0:
                optimizer.set_prior(prior)

            return optimizer

    def _finalize_train_method(self, loss, optimizer, global_step):
        """ Prepares nodes for training the neural network using an optimizer.

        :param loss: node for the desired loss function to minimize during training
        """
        with tf.name_scope('train'):
            trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            train_step = optimizer.minimize(loss, global_step=global_step, var_list=trainables)

            # DON'T put the nodes in there before the minimize call!
            # only after minimize was .._apply_dense() called and the nodes are ready
            self.summary_nodes['train_step'] = train_step

    def set_loss_function(self, loss_name):
        """ Set the loss function to minimize when optimizing.

        Note that the loss node can be obtained via :method:`neuralnetwork.get`.
        For evaluation it needs to be given to a tensorflow.Session.run() which
        will return the evaluated node given a dataset.

        :param loss_name: name of the loss function
        :return: loss node for :method:`tensorflow.train`
        """
        if loss_name not in self.loss_nodes:
            raise NotImplementedError
        with tf.name_scope('total'):
            loss = self.loss_nodes[loss_name]
            tf.summary.scalar('loss', loss)
            self.summary_nodes["loss"] = loss
        return loss

    def add_losses(self, y, y_):
        """ Add nodes to the graph to calculate losses for the dataset.

        :param y: predicted labels
        :param y_: true labels
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

    def add_output_layer(self, current_hidden_layer, hidden_out_dimension,
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
        y = self._nn_layer(current_hidden_layer, hidden_out_dimension, output_dimension, 'output',
                           trainables_collection=trainables_collection,
                           seed=seed, act=activation)
        self.summary_nodes['y'] = y
        logging.debug("y is " + str(y.get_shape()))
        return y

    def add_hidden_layers(self,input_layer, input_dimension, layer_dimensions, keep_prob = None,
                          activation=tf.nn.relu, trainables_collection=None, seed=None):
        """ Add fully connected hidden layers each with an additional dropout layer
         (makes the network robust against overfitting).

        The additional dropped layer will randomly drop samples according to the
         keep probability, i.e. 1 means all samples are keppt, 0 means all samples
         are dropped.

        :param input_layer: reference to the input layer
        :param input_dimension: number of nodes in `input_layer`
        :param keep_prob: reference to the placeholder with the *keep probability* for the dropped layer
        :param layer_dimensions: list of ints giving the number of nodes of each hidden layer
        :param activation: activation function of the hidden layers
        :param trainables_collection: specific collection to gather all weights of this layer
        :param seed: random number see to use for weights (seed is increased by one per layer)
        :return: reference to the last layer created
        """
        current_seed = seed
        last_layer = input_layer
        out_dimension = input_dimension
        for i in range(len(layer_dimensions)):
            if seed is not None:
                current_seed = seed+i
            number = str(i + 1)
            in_dimension = out_dimension
            out_dimension = layer_dimensions[i]
            layer_name = "layer" + number
            current_hidden = self._nn_layer(last_layer, in_dimension, out_dimension, layer_name,
                                            trainables_collection=trainables_collection,
                                            seed=current_seed, act=activation)
            logging.debug(layer_name + " is " + str(current_hidden.get_shape()))

            if keep_prob is not None:
                with tf.name_scope('dropout'):
                    last_layer = tf.nn.dropout(current_hidden, keep_prob)
            else:
                last_layer = current_hidden
            logging.debug("dropped" + number + " is " + str(last_layer.get_shape()))

        return last_layer

    def add_keep_probability(self):
        """ Adds a placeholder node for the keep probability of dropped layers.

        See :method:`neuralnetwork.add_hidden_layers`

        :return: reference to created node
        """
        keep_prob = tf.placeholder(dds_basetype, name="keep_probability")
        self.placeholder_nodes['keep_prob'] = keep_prob
        with tf.name_scope('dropout'):
            tf.summary.scalar('dropout_keep_probability', keep_prob)
        return keep_prob

    def add_writers(self, sess, log_dir):
        """ Adds log writers.

        Logs allow to visualize and debug the computational graph using
        TensorBoard (part of the Tensorflow package). Logs are files written
        to disk that contain all summary information.

        :param sess: Tensorflow Session
        :param log_dir: string giving directory to write files to
        """
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        self.summary_nodes["train_writer"] = train_writer
        test_writer = tf.summary.FileWriter(log_dir + '/test')
        self.summary_nodes["test_writer"] = test_writer
        
    @staticmethod
    def init_graph(sess):
        """ Initializes global variables in the computational graph.

        :param sess: Tensorflow Session
        """
        logging.debug ("Initializing global variables")
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()]
        )

    @staticmethod
    def weight_variable(shape, seed=None):
        """Create a weight variable, uniform randomly initialized in [-0.5, 0.5].

        :param shape: shape of the weight tensor to create
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
    def variable_summaries(var):
        """ Attach a lot of summaries (mean, stddev, min, max) to a given tensor
        for TensorBoard visualization.

        :param var: ref to the tensor variable to summarize
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

    def _nn_layer(self,
                  input_tensor, input_dim, output_dim,
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
        logging.info("Creating nn layer %s in scope %s with %d, %d" \
                     % (layer_name, scope_name, input_dim, output_dim))
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim], seed)
                tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)
                if trainables_collection is not None:
                    tf.add_to_collection(trainables_collection, weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
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
    def get_activations():
        """ Returns a dictionary with all known activation functions

        :return: dictionary with activations
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


