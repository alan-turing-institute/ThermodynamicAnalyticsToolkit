import tensorflow as tf
from sgldsampler import SGLDSampler as sgld
from SGLDMomentumSampler import SGLDMomentumSampler as sgld_momentum


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
    
    placeholder_nodes = {}
    """Lookup dictionary for input nodes, in TensorFlow parlance called placeholders."""
    summary_nodes = {}
    """Lookup dictionary for summary nodes, specific to TensorFlow. """


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
            return None

    def create(self, input_layer,
               input_dimension, layer_dimensions, output_dimension,
               optimizer, seed=None,
               add_dropped_layer=False):
        """ Creates the neural network model according to the specifications.

        The `input_layer` needs to be given along with its input_dimension.
        The output_layer needs to be specified here as the summaries and
        loss functions depend on them.

        :param input_layer: the input_layer
        :param input_dimension: number of nodes in `input_layer`
        :param layer_dimensions: a list of ints giving the number of nodes for
            each hidden layer.
        :param output_dimension: the number of nodes in the output layer
        :param optimizer: name of optimizer to use
        :param seed: seed for reproducible random values
        :param add_dropped_layer: whether to add dropped layer or not to protect against overfitting
        """
        self.summary_nodes.clear()
        if seed is not None:
            tf.set_random_seed(seed)

        y_ = self.add_true_labels(output_dimension)
        if add_dropped_layer:
            keep_prob = self.add_keep_probability()
        else:
            keep_prob = None
        if layer_dimensions is not None and len(layer_dimensions) != 0:
            last_hidden_layer = \
                self.add_hidden_layers(input_layer, input_dimension,
                                       layer_dimensions, keep_prob)
            y = self.add_output_layer(last_hidden_layer, layer_dimensions[-1], output_dimension)
        else:
            y = self.add_output_layer(input_layer, input_dimension, output_dimension)

        # print ("Creating summaries")
        loss = self.add_loss_summary(y, y_)
        self.add_accuracy_summary(y, y_)
        merged = tf.summary.merge_all()  # Merge all the summaries
        self.summary_nodes['merged'] = merged

        self.add_train_method(loss, optimizer, seed)

    def add_true_labels(self, output_dimension):
        """ Adds the known labels as placeholder nodes to the graph.

        :param output_dimension: number of output nodes
        :return: reference to created output layer
        """
        y_ = tf.placeholder(tf.float32, [None, output_dimension], name='y-input')
        # print("y_ is "+str(y_.get_shape()))
        self.placeholder_nodes['y_'] = y_
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
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.summary_nodes['accuracy'] = accuracy
        tf.summary.scalar('accuracy', accuracy)

    def add_train_method(self, loss, optimizer, seed):
        """ Adds nodes for training the neural network.

        :param loss: node for the desired loss function to minimize during training
        :param optimizer: name of the optimizer method, e.g. GradientDescent
        :param seed: seed value for the random number generator to obtain reproducible runs
        """
        with tf.name_scope('train'):
            # DON'T add placeholders only sometimes, e.g. when only a specific sampler
            # requires it. Always add them and only sometimes use them!
            step_width = tf.placeholder(tf.float32)
            tf.summary.scalar('step_width', step_width)
            self.placeholder_nodes['step_width'] = step_width
            inverse_temperature = tf.placeholder(tf.float32)
            tf.summary.scalar('inverse_temperature', inverse_temperature)
            self.placeholder_nodes['inverse_temperature'] = inverse_temperature
            friction_constant = tf.placeholder(tf.float32)
            tf.summary.scalar('friction_constant', friction_constant)
            self.placeholder_nodes['friction_constant'] = friction_constant

            global_step = tf.Variable(0, trainable=False)
            self.summary_nodes['global_step'] = global_step
            if optimizer == "StochasticGradientLangevinDynamics":
                sampler = sgld(step_width, inverse_temperature, seed=seed)
            elif optimizer == "StochasticMomentumLangevin":
                sampler = sgld_momentum(step_width, inverse_temperature, friction_constant, seed=seed)
            else:
                raise NotImplementedError("Unknown optimizer")
            train_step = sampler.minimize(loss, global_step=global_step)
            # DON'T put the nodes in there before the minimize call!
            # only after minimize was .._apply_dense() called and the nodes are ready
            if optimizer in ["StochasticGradientLangevinDynamics", "StochasticMomentumLangevin"]:
                self.summary_nodes['random_noise'] = sampler.random_noise
                self.summary_nodes['scaled_gradient'] = sampler.scaled_gradient
                self.summary_nodes['scaled_noise'] = sampler.scaled_noise
                self.summary_nodes['train_step'] = train_step
            if optimizer == "StochasticMomentumLangevin":
                self.summary_nodes['scaled_momentum'] = sampler.scaled_momentum
                self.summary_nodes['momentum'] = sampler.momentum

    def add_loss_summary(self, y, y_):
        """ Add nodes to the graph to calculate a loss for the dataset.

        The loss is the root mean squared error of the difference between
        the predicted label and the true label. Also, the cross entropy is
        added as a possible loss node.

        Note that the loss node can be obtained via :method:`neuralnetwork.get`.
        For evaluation it needs to be given to a tensorflow.Session.run() which
        will return the evaluated node given a dataset.

        :param y: predicted labels
        :param y_: true labels
        """
        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
            with tf.name_scope('total'):
                loss = tf.losses.mean_squared_error(labels=y_, predictions=y)
                self.summary_nodes['loss'] = loss
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('cross_entropy', cross_entropy)
        return loss

    def add_output_layer(self, current_hidden_layer, hidden_out_dimension, output_dimension):
        """ Add the output layer giving the predicted values.

        :param current_hidden_layer: last hidden layer which is to be connected to the output layer
        :param hidden_out_dimension: number of nodes in `current_hidden_layer`
        :param output_dimension: number of output nodes
        :return: reference to the output layer
        """
        y = self.nn_layer(current_hidden_layer, hidden_out_dimension, output_dimension, 'output', act=tf.nn.tanh)
        self.summary_nodes['y'] = y
        # print("y is " + str(y.get_shape()))
        return y

    def add_hidden_layers(self,input_layer, input_dimension, layer_dimensions, keep_prob = None):
        """ Add fully connected hidden layers each with an additional dropout layer
         (makes the network robust against overfitting).

        The additional dropped layer will randomly drop samples according to the
         keep probability, i.e. 1 means all samples are keppt, 0 means all samples
         are dropped.

        :param input_layer: reference to the input layer
        :param input_dimension: number of nodes in `input_layer`
        :param keep_prob: reference to the placeholder with the *keep probability* for the dropped layer
        :param layer_dimensions: list of ints giving the number of nodes of each hidden layer
        :return: reference to the last layer created
        """
        last_layer = input_layer
        out_dimension = input_dimension
        for i in range(len(layer_dimensions)):
            number = str(i + 1)
            in_dimension = out_dimension
            out_dimension = layer_dimensions[i]
            layer_name = "layer" + number
            current_hidden = self.nn_layer(last_layer, in_dimension, out_dimension, layer_name)
            # print(layer_name + " is " + str(current_hidden.get_shape()))

            if keep_prob is not None:
                with tf.name_scope('dropout'):
                    last_layer = tf.nn.dropout(current_hidden, keep_prob)
            else:
                last_layer = current_hidden
            # print("dropped" + number + " is " + str(last_layer.get_shape()))

        return last_layer

    def add_keep_probability(self):
        """ Adds a placeholder node for the keep probability of dropped layers.

        See :method:`neuralnetwork.add_hidden_layers`

        :return: reference to created node
        """
        keep_prob = tf.placeholder(tf.float32)
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
        # print ("Initializing global variables")
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()]
        )

    @staticmethod
    def weight_variable(shape):
        """Create a weight variable, uniform randomly initialized in [-0.5, 0.5].

        :param shape: shape of the weight tensor to create
        """
        initial = tf.random_uniform(shape, minval=-0.5, maxval=0.5)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """Create a bias variable with appropriate initialization.

        :param shape: shape of the weight tensor to create
        """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

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

    def nn_layer(self,
                 input_tensor, input_dim, output_dim,
                 layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.

        :param input_tensor: reference to input layer for this layer
        :param input_dim: number of nodes in `input_layer`
        :param output_dim: number of nodes in the created layer
        :param layer_name: created layer's name
        :param act: activation function to use for the nodes in the created layer
        :return: reference to the created layer
        """
        print("Creating nn layer %s with %d, %d" % (layer_name, input_dim, output_dim))
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights)
                weights_flat = tf.contrib.layers.flatten(weights)
                self.summary_nodes["weights"] = weights_flat
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                self.variable_summaries(biases)
                self.summary_nodes["biases"] = biases
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            if act == None:
                return preactivate
            else:
                activations = act(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations
