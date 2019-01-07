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
import time
from builtins import staticmethod
from math import sqrt, exp

import numpy as np
import pandas as pd
import scipy.sparse as sps
import tensorflow as tf
# scipy does not automatically import submodules
from scipy.sparse import linalg

from tensorflow.python.ops import variables

from TATi.common import create_input_layer, file_length, get_list_from_string, \
    initialize_config_map, setup_csv_file, setup_run_file, \
    setup_trajectory_file
from TATi.models.input.datasetpipeline import DatasetPipeline
from TATi.models.input.inmemorypipeline import InMemoryPipeline
from TATi.models.basetype import dds_basetype
from TATi.models.neuralnet_parameters import neuralnet_parameters
from TATi.models.neuralnetwork import NeuralNetwork
from TATi.models.trajectories.trajectorystate import TrajectoryState
from TATi.models.trajectories.trajectory_sampling_factory import TrajectorySamplingFactory
from TATi.models.trajectories.trajectory_training import TrajectoryTraining
from TATi.options.pythonoptions import PythonOptions

class model:
    """ This class combines the whole setup for creating a neural network.

    Moreover, it contains functions to either train or sample the loss function.

    """
    def __init__(self, FLAGS):
        # for allowing just reusing a new instance of this class, make sure
        # that we also reset the default graph before we start setting up
        # the neural network
        tf.reset_default_graph()

        self.FLAGS = FLAGS

        # reset trajectory instances
        self.trajectorystate = TrajectoryState(self)
        self.trajectory_sample = None
        self.trajectory_train = None
        self.reset_parameters(FLAGS)
        if "sampler" in self.FLAGS:
            self.trajectory_sample = TrajectorySamplingFactory.create(
                self.FLAGS.sampler, self.trajectorystate)
        if "optimizer" in self.FLAGS:
            self.trajectory_train = TrajectoryTraining(self.trajectorystate)

        self.number_of_parameters = 0  # number of biases and weights

        self.output_type = None
        self.scan_dataset_dimension()

        # mark input layer as to be created
        self.xinput = None
        self.x = None

        # mark already fixes variables
        self.fixed_variables = None

        # mark neuralnetwork, saver and session objects as to be created
        self.nn = None
        self.trainables = None
        self.true_labels = None
        self.saver = None
        self.sess = None

        # mark placeholder neuralnet_parameters as to be created (over walker)
        self.weights = []
        self.momenta_weights = []
        self.biases = []
        self.momenta_biases = []

        # mark placeholders for gradient and hessian computation as to be created
        self.gradients = None
        self.hessians = None

    def scan_dataset_dimension(self):
        if len(self.FLAGS.batch_data_files) > 0:
            self.input_dimension = self.FLAGS.input_dimension
            self.output_dimension = self.FLAGS.output_dimension
            try:
                self.FLAGS.add("dimension")
            except AttributeError:
                # add only on first call
                pass
            if self.FLAGS.batch_data_file_type == "csv":
                self.FLAGS.dimension = sum([file_length(filename)
                                            for filename in self.FLAGS.batch_data_files]) \
                                       - len(self.FLAGS.batch_data_files)
                if self.output_dimension == 1:
                    self.output_type = "binary_classification"  # labels in {-1,1}
                else:
                    self.output_type = "onehot_multi_classification"
            elif self.FLAGS.batch_data_file_type == "tfrecord":
                self.FLAGS.dimension = self._get_dimension_from_tfrecord(self.FLAGS.batch_data_files)
                self.output_type = "onehot_multi_classification"
            else:
                logging.info("Unknown file type")
                assert(0)
            self._check_valid_batch_size()

            logging.info("Parsing "+str(self.FLAGS.batch_data_files))

            self.number_of_parameters = 0 # number of biases and weights

    def init_input_pipeline(self):
        self.batch_next = self.create_input_pipeline(self.FLAGS)
        #self.input_pipeline.reset(self.sess)

    @staticmethod
    def _get_dimension_from_tfrecord(filenames):
        ''' Helper function to get the size of the dataset contained in a TFRecord.

        :param filenames: list of tfrecord files
        :return: total size of dataset
        '''
        dimension  = 0
        for filename in filenames:
            record_iterator = tf.python_io.tf_record_iterator(path=filename)
            for string_record in record_iterator:
                if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
                    example = tf.train.Example()
                    example.ParseFromString(string_record)
                    height = int(example.features.feature['height']
                                 .int64_list
                                 .value[0])

                    width = int(example.features.feature['width']
                                .int64_list
                                .value[0])
                    #logging.debug("height is "+str(height)+" and width is "+str(width))
                dimension += 1

        logging.info("Scanned " + str(dimension) + " records in tfrecord file.")

        return dimension

    def _check_valid_batch_size(self):
        ''' helper function to check that batch_size does not exceed dimension
        of dataset. After which it will be valid.

        :return: True - is smaller or equal, False - exceeded and capped batch_size
        '''
        if self.FLAGS.batch_size is None:
            logging.info("batch_size not set, setting to dimension of dataset.")
            self.FLAGS.batch_size = self.FLAGS.dimension
            return True
        if self.FLAGS.batch_size > self.FLAGS.dimension:
            logging.warning(" batch_size exceeds number of data items, capping.")
            self.FLAGS.batch_size = self.FLAGS.dimension
            return False
        else:
            return True

    def provide_data(self, features, labels, shuffle=False):
        ''' This function allows to provide an in-memory dataset using the Python
        API.

        :param features: feature part of dataset
        :param labels: label part of dataset
        :param shuffle: whether to shuffle the dataset initially or not
        '''
        logging.info("Using in-memory pipeline")
        self.input_dimension = len(features[0])
        self.output_dimension = len(labels[0])
        if self.output_dimension == 1:
            self.output_type = "binary_classification"  # labels in {-1,1}
        else:
            self.output_type = "onehot_multi_classification"
        assert( len(features) == len(labels) )
        try:
            self.FLAGS.dimension
        except AttributeError:
            self.FLAGS.add("dimension")
        self.FLAGS.dimension = len(features)
        self._check_valid_batch_size()
        self.input_pipeline = InMemoryPipeline(dataset=[features, labels],
                                               batch_size=self.FLAGS.batch_size,
                                               max_steps=self.FLAGS.max_steps,
                                               shuffle=shuffle, seed=self.FLAGS.seed)

    def create_input_pipeline(self, FLAGS, shuffle=False):
        """ This creates an input pipeline using the tf.Dataset module.

        :param FLAGS: parameters
        :param shuffle: whether to shuffle dataset or not
        """
        if FLAGS.in_memory_pipeline:
            logging.debug("Using in-memory pipeline")
            # create a session, parse the tfrecords with batch_size equal to dimension
            input_pipeline = DatasetPipeline(
                filenames=FLAGS.batch_data_files, filetype=FLAGS.batch_data_file_type,
                batch_size=FLAGS.dimension, dimension=FLAGS.dimension, max_steps=1,
                input_dimension=FLAGS.input_dimension, output_dimension=FLAGS.output_dimension,
                shuffle=shuffle, seed=FLAGS.seed)
            with tf.Session() as session:
                session.run(input_pipeline.iterator.initializer)
                xs, ys = input_pipeline.next_batch(session)

            self.input_pipeline = InMemoryPipeline(dataset=[xs,ys], batch_size=FLAGS.batch_size,
                                                   max_steps=FLAGS.max_steps,
                                                   shuffle=shuffle, seed=FLAGS.seed)
        else:
            logging.debug("Using tf.Dataset pipeline")
            self.input_pipeline = DatasetPipeline(filenames=FLAGS.batch_data_files, filetype=FLAGS.batch_data_file_type,
                                                  batch_size=FLAGS.batch_size, dimension=FLAGS.dimension, max_steps=FLAGS.max_steps,
                                                  input_dimension=self.input_dimension, output_dimension=self.output_dimension,
                                                  shuffle=shuffle, seed=FLAGS.seed)

    def get_averages_header(self, setup):
        if setup == "sample":
            return self.trajectory_sample.get_averages_header()
        elif setup == "train":
            return self.trajectory_train.get_averages_header()

    def get_run_header(self, setup):
        if setup == "sample":
            return self.trajectory_sample.get_run_header()
        elif setup == "train":
            return self.trajectory_train.get_run_header()

    def get_parameters(self):
        """ Getter for the internal set oF FLAGS controlling training and sampling.

        :return: FLAGS parameter set
        """
        return self.FLAGS

    def reset_parameters(self, FLAGS):
        """ Use to pass a different set of FLAGS controlling training or sampling.

        :param FLAGS: new set of parameters
        """
        self.FLAGS = FLAGS
        if self.trajectory_sample is not None:
            self.trajectory_sample.config_map = initialize_config_map()
        if self.trajectory_train is not None:
            self.trajectory_train.config_map = initialize_config_map()

        try:
            self.FLAGS.max_steps
        except KeyError:
            self.FLAGS.add("max_steps")
            self.FLAGS.max_steps = 1
        self.trajectorystate.FLAGS = FLAGS

    def create_model_file(self, initial_step, parameters, model_filename):
        self.assign_current_step(initial_step)
        self.assign_neural_network_parameters(parameters)
        self.save_model(model_filename)

    @staticmethod
    def setup_parameters(*args, **kwargs):
            return PythonOptions(add_keys=True, value_dict=kwargs)

    def reset_dataset(self):
        """ Re-initializes the dataset for a new run
        """
        self.input_pipeline.reset(self.sess)

    @staticmethod
    def _split_collection_per_walker(_collection, number_walkers):
        """ Helper function to split WEIGHTS and BIASES collection from
        tensorflow into weights and biases per walker.

        :param _collection: collection to split
        :param number_walkers: number of walkers to look for
        :return: list of split up collections
        """
        split_collection = []
        for i in range(number_walkers):
            split_collection.append([])
            scope_name = 'walker'+str(i+1)+'/'
            for var in _collection:
                if scope_name in var.name:
                    split_collection[-1].append(var)
        return split_collection

    def init_input_layer(self):
        # create input layer
        if self.xinput is None or self.x is None:
            input_columns = get_list_from_string(self.FLAGS.input_columns)
            self.xinput, self.x = create_input_layer(self.input_dimension, input_columns)

    def init_neural_network(self):
        if self.nn is None:
            self.nn = []
            self.loss = []
            self.trainables = []
            self.true_labels = NeuralNetwork.add_true_labels(self.output_dimension)

            # construct network per walker
            for i in range(self.FLAGS.number_walkers):
                with tf.name_scope('walker'+str(i+1)):
                    self.trainables.append('trainables_walker'+str(i+1))
                    self.nn.append(NeuralNetwork())
                    self.nn[-1].placeholder_nodes['y_'] = self.true_labels
                    keep_prob_node = self.nn[-1].add_keep_probability()
                    keep_prob = None if self.FLAGS.dropout is None else keep_prob_node
                    activations = NeuralNetwork.get_activations()
                    if self.FLAGS.seed is not None:
                        walker_seed = self.FLAGS.seed+i
                    else:
                        walker_seed = self.FLAGS.seed
                    self.loss.append(self.nn[-1].create(
                        self.x, self.FLAGS.hidden_dimension, self.output_dimension,
                        labels=self.true_labels,
                        trainables_collection=self.trainables[-1],
                        seed=walker_seed,
                        keep_prob=keep_prob,
                        hidden_activation=activations[self.FLAGS.hidden_activation],
                        output_activation=activations[self.FLAGS.output_activation],
                        loss_name=self.FLAGS.loss
                    ))
                    self.nn[-1].summary_nodes["accuracy"] = NeuralNetwork.add_accuracy_summary(
                        self.nn[-1].placeholder_nodes["y"],
                        self.nn[-1].placeholder_nodes["y_"],
                        self.output_type)
        else:
            self.loss = []
            for i in range(self.FLAGS.number_walkers):
                self.loss.append(self.nn[i].get_list_of_nodes(["loss"])[0])

    def fix_variables(self):
        fixed_variables = []
        values = None
        if self.fixed_variables is None:
            self.fixed_variables = {}
            # fix parameters
            if self.FLAGS.fix_parameters is not None:
                names, values = self.split_parameters_as_names_values(self.FLAGS.fix_parameters)
                fixed_variables = self.fix_parameters(names)
                logging.info("Excluded the following degrees of freedom: " + str(fixed_variables))
                logging.debug("Fixed vars are: " + str(self.fixed_variables))

                # additionally exclude fixed degrees from trainables_per_walker sets
                for i in range(self.FLAGS.number_walkers):
                    name_scope = 'walker' + str(i + 1)
                    with tf.name_scope(name_scope):
                        trainables = tf.get_collection_ref(self.trainables[i])
                        for var in fixed_variables:
                            removed_vars = model._fix_parameter_in_collection(trainables, var, name_scope+"'s trainables")
                            # make sure we remove one per walker
                            if len(removed_vars) != 1:
                                raise ValueError(
                                    "Cannot find " + var + " in walker " + str(i) + "." +
                                    " Have you checked the spelling, e.g., output/biases/Variable:0?")
                        logging.debug("Remaining trainable variables in walker " + str(i + 1)
                                      + ": " + str(tf.get_collection_ref(self.trainables[i])))
        else:
            if self.FLAGS.fix_parameters is not None:
                names, values = self.split_parameters_as_names_values(self.FLAGS.fix_parameters)
                fixed_variables.extend(self.fix_parameters(names))
                logging.info("Excluded the following degrees of freedom: " + str(fixed_variables))
        return fixed_variables, values


    def init_vectorized_gradients(self, add_vectorized_gradients):
        all_vectorized_gradients = []
        if self.gradients is None:
            self.gradients = []
            # construct (vectorized) gradient nodes and hessians
            for i in range(self.FLAGS.number_walkers):
                vectorized_gradients = []
                with tf.name_scope('walker' + str(i + 1)):
                    if self.FLAGS.do_hessians or add_vectorized_gradients:
                        # create node for gradient and hessian computation only if specifically
                        # requested as the creation along is costly (apart from the expensive part
                        # of evaluating the nodes eventually). This would otherwise slow down
                        # startup quite a bit even when hessians are not evaluated.
                        #print("GRADIENTS")
                        trainables = tf.get_collection_ref(self.trainables[i])
                        for tensor in trainables:
                            grad = tf.gradients(self.loss, tensor)
                            #print(grad)
                            vectorized_gradients.append(tf.reshape(grad, [-1]))
                        self.gradients.append(tf.reshape(tf.concat(vectorized_gradients, axis=0), [-1]))
                all_vectorized_gradients.append(vectorized_gradients)
        return all_vectorized_gradients

    def init_hessians(self, all_vectorized_gradients):
        if self.hessians is None:
            self.hessians = []
            for i in range(self.FLAGS.number_walkers):
                if self.FLAGS.do_hessians:
                    #print("HESSIAN")
                    total_dofs = 0
                    hessians = []
                    trainables = tf.get_collection_ref(self.trainables[i])
                    for gradient in all_vectorized_gradients[i]:
                        dofs = int(np.cumprod(gradient.shape))
                        total_dofs += dofs
                        #print(dofs)
                        # tensorflow cannot compute the gradient of a multi-dimensional mapping
                        # only of functions (i.e. one-dimensional output). Hence, we have to
                        # split the gradients into its components and do gradient on each
                        split_gradient = tf.split(gradient, num_or_size_splits=dofs)
                        for splitgrad in split_gradient:
                            for othertensor in trainables:
                                grad = tf.gradients(splitgrad, othertensor)
                                hessians.append(
                                    tf.reshape(grad, [-1]))
                    self.hessians.append(tf.reshape(tf.concat(hessians, axis=0), [total_dofs, total_dofs]))

    def get_split_weights_and_biases(self):
        # set number of degrees of freedom
        split_weights = self._split_collection_per_walker(
            tf.get_collection_ref(tf.GraphKeys.WEIGHTS), self.FLAGS.number_walkers)
        split_biases = self._split_collection_per_walker(
            tf.get_collection_ref(tf.GraphKeys.BIASES), self.FLAGS.number_walkers)
        self.number_of_parameters = \
            neuralnet_parameters.get_total_dof_from_list(split_weights[0]) \
            + neuralnet_parameters.get_total_dof_from_list(split_biases[0])
        logging.info("Number of dof per walker: "+str(self.number_of_parameters))
        return split_weights, split_biases

    def init_prior(self):
        # setup priors
        prior = {}
        try:
            if self.FLAGS.prior_factor is not None:
                prior["factor"] = self.FLAGS.prior_factor
            if self.FLAGS.prior_lower_boundary is not None:
                prior["lower_boundary"] = self.FLAGS.prior_lower_boundary
            if self.FLAGS.prior_power is not None:
                prior["power"] = self.FLAGS.prior_power
            if self.FLAGS.prior_upper_boundary is not None:
                prior["upper_boundary"] = self.FLAGS.prior_upper_boundary
        except AttributeError:
            pass
        return prior

    def init_model_save_restore(self):
        # setup model saving/recovering
        if self.saver is None:
            self.saver = tf.train.Saver(tf.get_collection_ref(tf.GraphKeys.WEIGHTS) +
                                   tf.get_collection_ref(tf.GraphKeys.BIASES) + \
                                   tf.get_collection_ref("Variables_to_Save"))

        # merge summaries at very end
        self.trajectorystate.summary = tf.summary.merge_all()  # Merge all the summaries

    def init_session(self):
        if self.sess is None:
            logging.debug("Using %s, %s threads " % (str(self.FLAGS.intra_ops_threads), str(self.FLAGS.inter_ops_threads)))
            self.sess = tf.Session(
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=self.FLAGS.intra_ops_threads,
                    inter_op_parallelism_threads=self.FLAGS.inter_ops_threads))

    def init_weights_access(self, setup, split_weights):
        if len(self.weights) == 0:
            assert(len(self.momenta_weights) == 0 )
            assert( len(split_weights) == self.FLAGS.number_walkers )
            for i in range(self.FLAGS.number_walkers):
                self.weights.append(neuralnet_parameters(split_weights[i]))
                assert( self.weights[i].get_total_dof() == self.get_total_weight_dof() )

    def init_weights_momenta_access(self, setup, split_weights):
        if len(self.momenta_weights) == 0:
            if setup is not None and "sample" in setup:
                for i in range(self.FLAGS.number_walkers):
                    momenta_weights = []
                    for v in split_weights[i]:
                        skip_var = False
                        if self.fixed_variables is not None:
                            for key in self.fixed_variables.keys():
                                if key in v.name:
                                    for var in self.fixed_variables[key]:
                                        if v.name == var.name:
                                            skip_var = True
                                            break
                        if not skip_var:
                            momenta_weights.append(self.trajectory_sample.sampler[i].get_slot(v, "momentum"))
                    #logging.debug("Momenta weights: "+str(momenta_weights))
                    if len(momenta_weights) > 0 and momenta_weights[0] is not None:
                        self.momenta_weights.append(neuralnet_parameters(momenta_weights))
                    else:
                        self.momenta_weights.append(None)

    def init_biases_access(self, setup, split_biases):
        if len(self.biases) == 0:
            assert( len(self.momenta_biases) == 0 )
            assert( len(split_biases) == self.FLAGS.number_walkers )
            for i in range(self.FLAGS.number_walkers):
                self.biases.append(neuralnet_parameters(split_biases[i]))
                assert (self.biases[i].get_total_dof() == self.get_total_bias_dof())

    def init_biases_momenta_access(self, setup, split_biases):
        if len(self.momenta_biases) == 0:
            if setup is not None and "sample" in setup:
                for i in range(self.FLAGS.number_walkers):
                    momenta_biases = []
                    for v in split_biases[i]:
                        skip_var = False
                        for key in self.fixed_variables.keys():
                            if key in v.name:
                                for var in self.fixed_variables[key]:
                                    if v.name == var.name:
                                        skip_var = True
                                        break
                        if not skip_var:
                            momenta_biases.append(self.trajectory_sample.sampler[i].get_slot(v, "momentum"))
                    #logging.debug("Momenta biases: "+str(momenta_biases))
                    if len(momenta_biases) > 0 and momenta_biases[0] is not None:
                        self.momenta_biases.append(neuralnet_parameters(momenta_biases))
                    else:
                        self.momenta_biases.append(None)

    def init_assign_fixed_parameters(self, fixed_variables, values):
        fix_parameter_assigns = None
        if self.FLAGS.fix_parameters is not None:
            all_values = []
            all_variables = []
            for i in range(len(fixed_variables)):
                var_name = fixed_variables[i]
                # skip None entries
                if var_name is None:
                    continue
                logging.debug("Trying to assign the fixed variable "+str(var_name))
                if var_name in self.fixed_variables.keys():
                    all_variables.extend(self.fixed_variables[var_name])
                    all_values.extend([values[i]]*len(self.fixed_variables[var_name]))
                else:
                    logging.warning("Could not assign "+var_name+" a value as it was not found before.")
            fix_parameter_assigns = self.create_assign_parameters(all_variables, all_values)
        return fix_parameter_assigns


    def restore_model(self, filename):
        # assign state of model from file if given
        if filename is not None:
            # Tensorflow DOCU says: initializing is not needed when restoring
            # however, global_variables are missing otherwise for storing kinetic, ...
            # tf.reset_default_graph()

            restore_path = filename.replace('.meta', '')
            self.saver.restore(self.sess, restore_path)
            logging.info("Model restored from file: %s" % restore_path)

    def assign_parse_parameter_file(self):
        # assign parameters of NN from step in given file
        if self.FLAGS.parse_parameters_file is not None \
                and (self.FLAGS.parse_steps is not None and (len(self.FLAGS.parse_steps) > 0)):
            step=self.FLAGS.parse_steps[0]
            for i in range(self.FLAGS.number_walkers):
                self.assign_weights_and_biases_from_file(self.FLAGS.parse_parameters_file, step,
                                                         walker_index=i, do_check=True)

    def init_network(self, filename=None, setup=None,
                     add_vectorized_gradients=False):
        """ Initializes the graph, from a stored model if filename is not None.

        :param filename: name of file containing stored model
        :param setup: "sample", "train" or else to add nodes that trigger a
                single sampling or training step. Otherwise they are not added.
                init_network() can be called consecutively with both variants
                to add either type of node.
        :param add_vectorized_gradients: add nodes to return gradients in fully
                vectorized form, i.e. in the same sequence as nn_weights and
                nn_biases parameters combined, see self.gradients.
        """
        # dataset was provided
        assert (self.input_dimension is not None)

        self.init_input_layer()
        self.init_neural_network()

        fixed_variables, values = self.fix_variables()
        logging.debug("Remaining global trainable variables: " + str(variables.trainable_variables()))

        all_vectorized_gradients = self.init_vectorized_gradients(add_vectorized_gradients)
        self.init_hessians(all_vectorized_gradients)

        split_weights, split_biases = self.get_split_weights_and_biases()

        prior = self.init_prior()
        self.trajectorystate.init_trajectory(self)
        if setup is not None and "sample" in setup:
            self.trajectory_sample.init_trajectory(prior, self)
        if setup is not None and "train" in setup:
            self.trajectory_train.init_trajectory(prior, self)
        self.trajectorystate.init_step_placeholder()
        self.trajectorystate.init_parse_directions()

        self.init_model_save_restore()

        self.init_weights_access(setup, split_weights)
        if setup is not None and "sample" in setup:
            self.init_weights_momenta_access(setup, split_weights)
        self.init_biases_access(setup, split_biases)
        if setup is not None and "sample" in setup:
            self.init_biases_momenta_access(setup, split_biases)

        fix_parameter_assigns = self.init_assign_fixed_parameters(fixed_variables, values)

        self.init_session()

        ### Now the session object is created, graph must be done here!

        # initialize constants in graph
        NeuralNetwork.init_graph(self.sess)

        # initialize dataset
        #self.input_pipeline.reset(self.sess)

        # run assigns for fixed parameters
        if self.FLAGS.fix_parameters is not None:
            logging.debug("Assigning the following values to fixed degrees of freedom: "+str(values))
            self.sess.run(fix_parameter_assigns)

        self.restore_model(filename)

        self.assign_parse_parameter_file()

    def get_total_weight_dof(self):
        return self.weights[0].get_total_dof()

    def get_total_bias_dof(self):
        return self.biases[0].get_total_dof()

    def compute_optimal_stepwidth(self, walker_index=0):
        assert( walker_index < self.FLAGS.number_walkers )
        placeholder_nodes = self.nn[walker_index].get_dict_of_nodes(["learning_rate", "y_"])

        # get first batch of data
        self.reset_dataset()
        features, labels = self.input_pipeline.next_batch(self.sess)

        # place in feed dict
        feed_dict = {
            self.xinput: features,
            placeholder_nodes["y_"]: labels,
            placeholder_nodes["learning_rate"]: self.FLAGS.learning_rate
        }
        if self.FLAGS.dropout is not None:
            feed_dict.update({placeholder_nodes["keep_prob"] : self.FLAGS.dropout})

        hessian_eval = self.sess.run(self.hessians[walker_index], feed_dict=feed_dict)
        lambdas, _ = sps.linalg.eigs(hessian_eval, k=1)
        optimal_step_width = 2/sqrt(lambdas[0])
        logging.info("Optimal step width would be "+str(optimal_step_width))

    def save_model(self, filename):
        """ Saves the current neural network model to a set of files,
        whose prefix is given by filename.

        :param filename: prefix of set of model files
        :return: path where model was saved
        """
        return self.saver.save(self.sess, filename)

    def finish(self):
        """ Closes all open files and saves the model if desired
        """
        try:
            if self.FLAGS.save_model is not None:
                save_path = self.save_model(self.FLAGS.save_model.replace('.meta', ''))
                logging.debug("Model saved in file: %s" % save_path)
        except AttributeError:
            pass

    @staticmethod
    def _find_all_in_collections(_collection, _name):
        """ Helper function to return all indices of variables in a collection
         that match with the given `_name`. Note that this removes possible
         walker name scopes.

        :param _collection: collection to search through
        :param _name: tensor/variable name to look for
        :return: list of matching indices
        """
        variable_indices = []
        for i in range(len(_collection)):
            target_name = _collection[i].name
            walker_target_name = target_name[target_name.find("/")+1:]
            logging.debug("Comparing %s to %s and %s" % (_name, target_name, walker_target_name))
            if target_name == _name or walker_target_name == _name:
                variable_indices.append(i)
        return variable_indices

    @staticmethod
    def _extract_from_collections(_collection, _indices):
        """ Helper function to remove all elements associated to each index
        in `indices` from `collections`, gathering them in a list that is
        returned

        :param _collection: collection to remove elements from
        :param _indices: list of indices to extract
        :return: list of elements removed from collection
        """
        variables = []
        _indices.sort(reverse=True)
        for i in _indices:
            variables.append(_collection[i])
            del _collection[i]
        return variables

    @staticmethod
    def _fix_parameter_in_collection(_collection, _name, _collection_name="collection"):
        """ Allows to fix a parameter (not modified during optimization
        or sampling) by removing the first instance named _name from trainables.

        :param _collection: (trainables or other) collection to remove parameter from
        :param _name: name of parameter to fix
        :return: None or Variable ref that was fixed
        """
        variable_indices = model._find_all_in_collections(_collection, _name)
        logging.debug("Indices matching in "+_collection_name+" with "
                     +_name+": "+str(variable_indices))
        fixed_variable = model._extract_from_collections(_collection, variable_indices)
        return fixed_variable

    @staticmethod
    def _fix_parameter(_name):
        """ Allows to fix a parameter (not modified during optimization
        or sampling) by removing the first instance named _name from trainables.

        :param _name: name of parameter to fix
        :return: None or Variable ref that was fixed
        """
        variable = None
        trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        if "weight" in _name:
            other_collection = tf.get_collection_ref(tf.GraphKeys.WEIGHTS)
            other_collection_name = "weights"
        elif "bias" in _name:
            other_collection = tf.get_collection_ref(tf.GraphKeys.BIASES)
            other_collection_name = "biases"
        else:
            logging.warning("Unknown parameter category for "+str(_name) \
                            +"), removing only from trainables.")

        trainable_variable = model._fix_parameter_in_collection(trainable_collection, _name, "trainables")
        variable = model._fix_parameter_in_collection(other_collection, _name, other_collection_name)

        #print(tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES))
        #if "weight" in _name:
        #    print(tf.get_collection_ref(tf.GraphKeys.WEIGHTS))
        #else:
        #    print(tf.get_collection_ref(tf.GraphKeys.BIASES))

        if trainable_variable == variable:
            return variable
        else:
            return None

    @staticmethod
    def _assign_parameter(_var, _value):
        """ Creates an assignment node, adding it to the graph.

        :param _var: tensorflow variable ref
        :param _value: value to assign to it, must have same shape
        :return: constant value node and assignment node
        """
        value_t = tf.constant(_value, dtype=_var.dtype)
        assign_t = _var.assign(value_t)
        return value_t, assign_t

    def fix_parameters(self, names):
        """ Fixes the parameters given by their names

        :param names: list of names
        :return: list of tensorflow variables that are fixed
        """
        retlist = []
        for name in names:
            logging.debug("Looking for variable %s to fix." % (name))
            # look for tensor in already fixed variables
            variable_list = None
            retvariable_list = self._fix_parameter(name)
            logging.debug("Updated fixed parameters by: "+str(retvariable_list))
            if retvariable_list is not None:
                for retvariable in retvariable_list:
                    if name in self.fixed_variables.keys():
                        self.fixed_variables[name].append(retvariable)
                    else:
                        self.fixed_variables[name] = [retvariable]
                if name in self.fixed_variables.keys():
                    retlist.append(name)
                else:
                    retlist.append(None)
        return retlist

    def assign_current_step(self, step, walker_index=0):
        assert( walker_index < self.FLAGS.number_walkers )
        # set step
        if ('global_step' in self.nn[walker_index].summary_nodes.keys()):
            sample_step_placeholder = self.trajectorystate.step_placeholder[walker_index]
            feed_dict = {sample_step_placeholder: step}
            set_step = self.sess.run(self.trajectorystate.global_step_assign_t[walker_index], feed_dict=feed_dict)
            assert (set_step == step)

    def assign_neural_network_parameters(self, parameters, walker_index=0):
        """ Assigns the parameters of the neural network from
        the given array.

        :param parameters: list of values, one for each weight and bias
        :param walker_index: index of the replicated network (in the graph)
        """
        weights_dof = self.weights[walker_index].get_total_dof()
        self.weights[walker_index].assign(self.sess, parameters[0:weights_dof])
        self.biases[walker_index].assign(self.sess, parameters[weights_dof:])

    def assign_weights_and_biases(self, weights_vals, biases_vals, walker_index=0, do_check=False):
        """ Assigns weights and biases of a neural network.

        :param weights_vals: flat weights parameters
        :param biases_vals: flat bias parameters
        :param walker_index: index of the replicated network (in the graph)
        :param do_check: whether to check set values (and print) or not
        :return evaluated weights and bias on do_check or None otherwise
        """
        if weights_vals.size > 0:
            self.weights[walker_index].assign(self.sess, weights_vals)
        if biases_vals.size > 0:
            self.biases[walker_index].assign(self.sess, biases_vals)

        # get the input and biases to check against what we set
        if do_check:
            weights_eval = self.weights[walker_index].evaluate(self.sess)
            biases_eval = self.biases[walker_index].evaluate(self.sess)
            logging.info("Evaluating walker #"+str(walker_index) \
                         +" at weights " + str(weights_eval[0:10]) \
                         + ", biases " + str(biases_eval[0:10]))
            assert( np.allclose(weights_eval, weights_vals, atol=1e-7) )
            assert( np.allclose(biases_eval, biases_vals, atol=1e-7) )
            return weights_eval, biases_eval
        return None

    def assign_weights_and_biases_from_dataframe(self, df_parameters, rownr, walker_index=0, do_check=False):
        """ Parse weight and bias values from a dataframe given a specific step
        to set the neural network's parameters.

        :param df_parameters: pandas dataframe
        :param rownr: rownr to set
        :param walker_index: index of the replicated network (in the graph)
        :param do_check: whether to evaluate (and print) set parameters
        :return evaluated weights and bias on do_check or None otherwise
        """
        # check that column names are in order
        weight_numbers = []
        bias_numbers = []
        for keyname in df_parameters.columns:
            if (keyname[1] >= "0" and keyname[1] <= "9"):
                if ("w" == keyname[0]):
                    weight_numbers.append(int(keyname[1:]))
                elif "b" == keyname[0]:
                    bias_numbers.append(int(keyname[1:]))
            else:
                if ("weight" in keyname):
                    weight_numbers.append(int(keyname[6:]))
                elif ("bias" in keyname):
                    bias_numbers.append(int(keyname[4:]))

        def get_start_index(numbers, df, paramlist):
            start_index = -1
            if len(numbers) > 0:
                for param in paramlist:
                    try:
                        start_index = df.columns.get_loc("%s%d" % (param, numbers[0]))
                        break
                    except KeyError:
                        pass
            return start_index

        weights_start = get_start_index(weight_numbers, df_parameters, ["weight", "w"])
        biases_start = get_start_index(bias_numbers, df_parameters, ["bias", "b"])

        def check_aligned(numbers):
            lastnr = None
            for nr in numbers:
                if lastnr is not None:
                    if lastnr >= nr:
                        break
                lastnr = nr
            return lastnr

        weights_aligned = (len(weight_numbers) > 0) and (check_aligned(weight_numbers) == weight_numbers[-1])
        biases_aligned = (len(bias_numbers) > 0) and (check_aligned(bias_numbers) == bias_numbers[-1])
        values_aligned = weights_aligned and biases_aligned and weights_start < biases_start

        if values_aligned:
            # copy values in one go
            weights_vals = df_parameters.iloc[rownr, weights_start:biases_start].values
            biases_vals = df_parameters.iloc[rownr, biases_start:].values
        else:
            # singly pick each value

            # create internal array to store parameters
            weights_vals = self.weights[walker_index].create_flat_vector()
            biases_vals = self.biases[walker_index].create_flat_vector()
            for keyname in df_parameters.columns:
                if (keyname[1] >= "0" and keyname[1] <= "9"):
                    if ("w" == keyname[0]):
                        weights_vals[int(keyname[1:])] = df_parameters.loc[rownr, [keyname]].values[0]
                    elif "b" == keyname[0]:
                        biases_vals[int(keyname[1:])] = df_parameters.loc[rownr, [keyname]].values[0]
                    else:
                        # not a parameter column
                        continue
                else:
                    if ("weight" in keyname):
                        weights_vals[int(keyname[6:])] = df_parameters.loc[rownr, [keyname]].values[0]
                    elif ("bias" in keyname):
                        biases_vals[int(keyname[4:])] = df_parameters.loc[rownr, [keyname]].values[0]

        logging.debug("Read row (first three weights and biases) "+str(rownr)+":"+str(weights_vals[:5]) \
                      +"..."+str(biases_vals[:5]))

        return self.assign_weights_and_biases(weights_vals, biases_vals, walker_index, do_check)

    def assign_weights_and_biases_from_file(self, filename, step, walker_index=0, do_check=False):
        """ Parse weight and bias values from a CSV file given a specific step
        to set the neural network's parameters.

        :param filename: filename to parse
        :param step: step to set (i.e. value in "step" column designates row)
        :param walker_index: index of the replicated network (in the graph)
        :param do_check: whether to evaluate (and print) set parameters
        :return evaluated weights and bias on do_check or None otherwise
        """
        # parse csv file
        df_parameters = pd.read_csv(filename, sep=',', header=0)
        if step in df_parameters.loc[:, 'step'].values:
            rowlist = np.where((df_parameters.loc[:, 'step'].values == step))[0]
            if self.FLAGS.number_walkers > 1:
                # check whether param files contains entries for multiple walkers
                id_idx = df_parameters.columns.get_loc("id")
                num_ids = df_parameters.iloc[rowlist,id_idx].max() - \
                          df_parameters.iloc[rowlist,id_idx].min() +1
                if num_ids >= self.FLAGS.number_walkers:
                    rowlist = np.where((df_parameters.iloc[rowlist,id_idx].values == walker_index))
                else:
                    logging.info("Not enough values in parse_parameters_file for all walkers, using first for all.")
            if len(rowlist) > 1:
                logging.warning("Found multiple matching entries to step "+str(step) \
                                +" and walker #"+str(walker_index))
            elif len(rowlist) == 0:
                raise ValueError("Step "+str(step)+" and walker #"+str(walker_index)+" not found.")
            rownr = rowlist[0]
            self.assign_current_step(step, walker_index=walker_index)
            return self.assign_weights_and_biases_from_dataframe(
                df_parameters=df_parameters,
                rownr=rownr,
                walker_index=walker_index,
                do_check=do_check
            )
        else:
            logging.debug("Step " + str(step) + " not found in file.")
            return None

    def create_assign_parameters(self, variables, values):
        """ Creates assignment operation for multiple parameters at once.

        :param variables: dict of tensorflow variable names and list of variable
                tensors
        :param values: list of values to assign to
        """
        logging.debug("Assigning to vars: "+str(variables))
        logging.debug("Assigning values :"+str(values))
        assert( len(variables) == len(values) )
        assigns=[]
        for i in range(len(variables)):
            value_t, assign_t = self._assign_parameter(
                variables[i],
                np.reshape(values[i], newshape=variables[i].shape))
            assigns.append(assign_t)

        return assigns

    @staticmethod
    def split_parameters_as_names_values(_string):
        """ Extracts parameter names and values from the given string in the form:
         name=value;name=value;...

        :param _string: string to tokenize
        """
        names=[]
        values=[]
        for a in _string.split(";"):
            if len(a) <= 1:
                continue
            b=a.split("=", 2)
            names.append(b[0])
            values.append(np.fromstring(b[1], dtype=float, sep=","))
        return names, values

    def sample(self, return_run_info = False, return_trajectories = False, return_averages = False):
        """ Performs the actual sampling of the neural network `nn` given a dataset `ds` and a
        Session `session`.

        :param return_run_info: if set to true, run information will be accumulated
                inside a numpy array and returned
        :param return_trajectories: if set to true, trajectories will be accumulated
                inside a numpy array and returned (adhering to FLAGS.every_nth)
        :param return_averages: if set to true, accumulated average values will be
                returned as numpy array
        :return: either thrice None or lists (per walker) of pandas dataframes
                depending on whether either parameter has evaluated to True
        """
        return self._execute_trajectory_run(self.trajectory_sample, self.sess,
                                     return_run_info, return_trajectories, return_averages)

    def train(self, return_run_info = False, return_trajectories = False, return_averages=False):
        """ Performs the actual training of the neural network `nn` given a dataset `ds` and a
        Session `session`.

        :param return_run_info: if set to true, run information will be accumulated
                inside a numpy array and returned
        :param return_trajectories: if set to true, trajectories will be accumulated
                inside a numpy array and returned (adhering to FLAGS.every_nth)
        :param return_averages: if set to true, accumulated average values will be
                returned as numpy array
        :return: either twice None or a pandas dataframe depending on whether either
                parameter has evaluated to True
        """
        return self._execute_trajectory_run(self.trajectory_train, self.sess,
                                     return_run_info, return_trajectories, return_averages)

    def _execute_trajectory_run(self, trajectory, session, return_run_info = False, return_trajectories = False, return_averages=False):
        retvals = trajectory.execute(session,
            { "input_pipeline": self.input_pipeline,
              "xinput": self.xinput,
              "true_labels": self.true_labels},
            return_run_info, return_trajectories, return_averages)
        self.finish()
        return retvals

    def deactivate_file_writing(self):
        """ Deactivates writing of average, run info and trajectory files.

        """
        for entity in [self.trajectory_train, self.trajectory_sample]:
            if entity is not None:
                entity.set_config_map("do_write_averages_file", False)
                entity.set_config_map("do_write_run_file", False)
                entity.set_config_map("do_write_trajectory_file", False)
