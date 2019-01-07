#
#    ThermodynamicAnalyticsToolkit - explore high-dimensional manifold of neural networks
#    Copyright (C) 2018 The University of Edinburgh
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

try:
    from tqdm import tqdm # allows progress bar
    tqdm_present = True
    # workaround: otherwise we get deadlock on exceptions,
    # see https://github.com/tqdm/tqdm/issues/469
    tqdm.monitor_interval = 0
except ImportError:
    tqdm_present = False

from tensorflow.python.ops import variables

from TATi.common import create_input_layer, file_length, get_list_from_string, \
    get_trajectory_header, initialize_config_map, setup_csv_file, setup_run_file, \
    setup_trajectory_file
from TATi.models.accumulators.averagesaccumulator import AveragesAccumulator
from TATi.models.accumulators.runinfoaccumulator import RuninfoAccumulator
from TATi.models.accumulators.trajectoryaccumulator import TrajectoryAccumulator
from TATi.models.accumulators.accumulated_values import AccumulatedValues
from TATi.models.input.datasetpipeline import DatasetPipeline
from TATi.models.input.inmemorypipeline import InMemoryPipeline
from TATi.models.basetype import dds_basetype
from TATi.models.neuralnet_parameters import neuralnet_parameters
from TATi.models.neuralnetwork import NeuralNetwork
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

        self.reset_parameters(FLAGS)

        self.number_of_parameters = 0   # number of biases and weights

        self.output_type = None
        self.scan_dataset_dimension()

        # mark input layer as to be created
        self.xinput = None
        self.x = None

        # mark resource variables as to be created
        self.resources_created = None

        # mark already fixes variables
        self.fixed_variables = None

        # mark neuralnetwork, saver and session objects as to be created
        self.nn = None
        self.trainables = None
        self.true_labels = None
        self.saver = None
        self.sess = None

        self.optimizer = None
        self.sampler = None

        # mark placeholder neuralnet_parameters as to be created (over walker)
        self.weights = []
        self.momenta_weights = []
        self.biases = []
        self.momenta_biases = []

        # mark placeholders for gradient and hessian computation as to be created
        self.gradients = None
        self.hessians = None

        # mark step assign op as to be created
        self.step_placeholder = None
        self.global_step_assign_t = None

        # mark writer as to be created
        self.averages_writer = None
        self.run_writer = None
        self.trajectory_writer = None
        self.directions = None

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
        self.config_map = initialize_config_map()

        try:
            self.FLAGS.max_steps
        except KeyError:
            self.FLAGS.add("max_steps")
            self.FLAGS.max_steps = 1

    def create_resource_variables(self):
        """ Creates some global resource variables to hold statistical quantities
        during sampling.
        """
        static_vars_float = ["current_kinetic", "kinetic_energy", \
                             "old_total_energy", "inertia", "momenta", "gradients", "virials", "noise"]
        static_vars_int64 = ["accepted", "rejected"]
        for i in range(self.FLAGS.number_walkers):
            with tf.variable_scope("var_walker"+str(i+1), reuse=self.resources_created):
                with tf.variable_scope("accumulate", reuse=self.resources_created):
                    for key in static_vars_float:
                        tf.get_variable(key, shape=[], trainable=False,
                                        initializer=tf.zeros_initializer,
                                        use_resource=True, dtype=dds_basetype)
                    for key in static_vars_int64:
                        # the following are used for HMC to measure rejection rate
                        tf.get_variable(key, shape=[], trainable=False,
                                        initializer=tf.zeros_initializer,
                                        use_resource=True, dtype=tf.int64)

        self.resources_created = True

    def get_config_map(self, key):
        if key in self.config_map.keys():
            return self.config_map[key]
        else:
            return None

    def set_config_map(self, key, value):
        self.config_map[key] = value

    def write_run_row(self, line):
        self.run_writer.writerow(line)

    def write_trajectory_row(self, line):
        self.trajectory_writer.writerow(line)

    def write_averages_row(self, line):
        self.averages_writer.writerow(line)

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

    def init_parse_directions(self):
        # directions span a subspace to project trajectories. This may be
        # used to not store overly many degrees of freedom per step.
        if self.FLAGS.directions_file is not None:
            try:
                # try without header
                self.directions = np.loadtxt(self.FLAGS.directions_file, delimiter=',', skiprows=0)
            except ValueError:
                # if it fails, skip header
                self.directions = np.loadtxt(self.FLAGS.directions_file, delimiter=',', skiprows=1)
            if len(self.directions.shape) == 1:
                self.directions = np.expand_dims(self.directions, axis=0)
        else:
            self.directions = None

    def init_train(self, setup, prior):
        # setup training/sampling
        if setup is not None:
            if "train" in setup:
                self.optimizer = []
                for i in range(self.FLAGS.number_walkers):
                    with tf.variable_scope("var_walker" + str(i + 1)):
                        self.optimizer.append(self.nn[i].add_train_method(
                            self.loss[i], optimizer_method=self.FLAGS.optimizer,
                            prior=prior))
            else:
                logging.info("Not adding train method.")

    def init_sample(self, setup, prior):
        # setup training/sampling
        if setup is not None:
            if "sample" in setup:
                self.sampler = []
                for i in range(self.FLAGS.number_walkers):
                    if self.FLAGS.seed is not None:
                        walker_seed = self.FLAGS.seed + i
                    else:
                        walker_seed = self.FLAGS.seed

                    # raise exception if HMC is used with multiple walkers
                    if "HamiltonianMonteCarlo" in self.FLAGS.sampler \
                        and self.FLAGS.number_walkers > 1:
                        raise NotImplementedError(
                            "HamiltonianMonteCarlo implementation has not been properly tested with multiple walkers.")

                    self.sampler.append(self.nn[i]._prepare_sampler(
                        self.loss[i], sampling_method=self.FLAGS.sampler,
                        seed=walker_seed, prior=prior,
                        sigma=self.FLAGS.sigma, sigmaA=self.FLAGS.sigmaA))
                # create gradients
                grads_and_vars = []
                for i in range(self.FLAGS.number_walkers):
                    with tf.name_scope('gradients_walker'+str(i+1)):
                        trainables = tf.get_collection_ref(self.trainables[i])
                        grads_and_vars.append(self.sampler[i].compute_and_check_gradients(
                            self.loss[i], var_list=trainables))

                # add position update nodes
                for i in range(self.FLAGS.number_walkers):
                    with tf.variable_scope("var_walker" + str(i + 1)):
                        global_step = self.nn[i]._prepare_global_step()
                        train_step = self.sampler[i].apply_gradients(
                            grads_and_vars, i, global_step=global_step,
                            name=self.sampler[i].get_name())
                    self.nn[i].summary_nodes['sample_step'] = train_step
                    self.nn[i].summary_nodes['EQN_step'] = self.sampler[i].EQN_update
            else:
                logging.info("Not adding sample method.")

    def init_step_placeholder(self, setup):
        if setup is not None:
            if "train" in setup or "sample" in setup:
                if self.step_placeholder is None:
                    self.step_placeholder = []
                    for i in range(self.FLAGS.number_walkers):
                        with tf.name_scope("walker"+str(i+1)):
                            self.step_placeholder.append(tf.placeholder(shape=(), dtype=tf.int32))
                if self.global_step_assign_t is None:
                    self.global_step_assign_t = []
                    for i in range(self.FLAGS.number_walkers):
                        with tf.name_scope("walker"+str(i+1)):
                            self.global_step_assign_t.append(tf.assign(self.nn[i].summary_nodes['global_step'], self.step_placeholder[i]))
            else:
                logging.debug("Not adding step placeholder or global step.")

    def init_model_save_restore(self):
        # setup model saving/recovering
        if self.saver is None:
            self.saver = tf.train.Saver(tf.get_collection_ref(tf.GraphKeys.WEIGHTS) +
                                   tf.get_collection_ref(tf.GraphKeys.BIASES) + \
                                   tf.get_collection_ref("Variables_to_Save"))

        # merge summaries at very end
        self.summary = tf.summary.merge_all()  # Merge all the summaries


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
                            momenta_weights.append(self.sampler[i].get_slot(v, "momentum"))
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
                            momenta_biases.append(self.sampler[i].get_slot(v, "momentum"))
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

        # create global variables, one for every walker in its replicated graph
        self.create_resource_variables()
        self.static_vars, self.zero_assigner, self.assigner, self.placeholders = \
            self._create_static_variable_dict(self.FLAGS.number_walkers)

        split_weights, split_biases = self.get_split_weights_and_biases()

        prior = self.init_prior()
        self.init_train(setup, prior)
        self.init_sample(setup, prior)
        self.init_step_placeholder(setup)

        self.init_model_save_restore()

        self.init_weights_access(setup, split_weights)
        self.init_biases_access(setup, split_biases)

        self.init_parse_directions()

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

    def init_files(self, setup):
        """ Initializes the output files.
        """
        header = None
        if setup == "sample":
            header = self.get_sample_header()
        elif setup == "train":
            header = self.get_train_header()

        try:
            if self.averages_writer is None:
                if self.FLAGS.averages_file is not None:
                    self.config_map["do_write_averages_file"] = True
                    self.averages_writer, self.config_map["averages_file"] = setup_csv_file(self.FLAGS.averages_file, self.get_averages_header(setup))
        except AttributeError:
            pass
        try:
            if self.run_writer is None:
                self.run_writer = setup_run_file(self.FLAGS.run_file, header, self.config_map)
        except AttributeError:
            pass
        try:
            if self.trajectory_writer is None:
                if self.directions is not None:
                    number_weights = self.directions.shape[0]
                    number_biases = 0
                else:
                    number_weights = self.weights[0].get_total_dof()
                    number_biases = self.biases[0].get_total_dof()
                self.trajectory_writer = setup_trajectory_file(self.FLAGS.trajectory_file,
                                                               number_weights,
                                                               number_biases,
                                                               self.config_map)
        except AttributeError:
            pass

    def get_averages_header(self, setup=None):
        """ Prepares the distinct header for the averages file for sampling

        :param setup: sample, train or None
        """
        header = ['id', 'step', 'epoch', 'loss']
        if setup == "train":
            if self.FLAGS.optimizer == "GradientDescent":
                header += ['average_virials']
        elif setup == "sample":
            if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                header += ['ensemble_average_loss', 'average_virials']
            elif self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder",
                                        "GeometricLangevinAlgorithm_2ndOrder",
                                        "BAOAB",
                                        "CovarianceControlledAdaptiveLangevinThermostat"]:
                header += ['ensemble_average_loss', 'average_kinetic_energy', 'average_virials', 'average_inertia']
            elif "HamiltonianMonteCarlo" in self.FLAGS.sampler:
                header += ['ensemble_average_loss', 'average_kinetic_energy', 'average_virials', 'average_inertia', 'average_rejection_rate']
        return header

    def get_sample_header(self):
        """ Prepares the distinct header for the run file for sampling
        """
        header = ['id', 'step', 'epoch', 'accuracy', 'loss', 'time_per_nth_step']
        if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
            header += ['scaled_gradient', 'virial', 'scaled_noise']
        elif self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder",
                                    "GeometricLangevinAlgorithm_2ndOrder",
                                    "BAOAB",
                                    "CovarianceControlledAdaptiveLangevinThermostat"]:
            header += ['total_energy', 'kinetic_energy', 'scaled_momentum',
                      'scaled_gradient', 'virial', 'scaled_noise']
        elif "HamiltonianMonteCarlo" in self.FLAGS.sampler:
            header += ['total_energy', 'old_total_energy', 'kinetic_energy', 'scaled_momentum',
                      'scaled_gradient', 'virial', 'average_rejection_rate']
        return header

    def get_train_header(self):
        """ Prepares the distinct header for the run file for training
        """
        return ['id', 'step', 'epoch', 'accuracy', 'loss', 'time_per_nth_step', 'scaled_gradient', 'virial']

    def get_total_weight_dof(self):
        return self.weights[0].get_total_dof()

    def get_total_bias_dof(self):
        return self.biases[0].get_total_dof()

    @staticmethod
    def _dict_append(_dict, _key, _item):
        if _key in _dict.keys():
            _dict[_key].append(_item)
        else:
            _dict[_key] = [_item]

    def _create_static_variable_dict(self, number_replicated_graphs):
        """ Instantiate all sampler's resource variables. Also create
        assign zero nodes.

        This returns a dictionary with lists as values where the lists contain
        the created variable for each replicated graph associated to a walker.

        :param number_replicated_graphs: number of replicated graphs to instantiate for
        :return: dict with static_var lists (one per walker), equivalent dict
                for zero assigners, dict with assigners (required for HMC), and dict
                with placeholders for assigners (required for HMC)
        """
        static_vars_float = ["current_kinetic", "kinetic_energy", \
                             "old_total_energy", "inertia", "momenta", "gradients", "virials", "noise"]
        static_vars_int64 = ["accepted", "rejected"]
        static_var_dict = {}
        zero_assigner_dict = {}
        assigner_dict = {}
        placeholder_dict = {}
        for i in range(number_replicated_graphs):
            with tf.variable_scope("var_walker" + str(i + 1), reuse=True):
                with tf.variable_scope("accumulate", reuse=True):
                    for key in static_vars_float:
                        static_var =  tf.get_variable(key, dtype=dds_basetype)
                        model._dict_append(static_var_dict, key, static_var)
                        zero_assigner = static_var.assign(0.)
                        model._dict_append(zero_assigner_dict, key, zero_assigner)
                    for key in static_vars_int64:
                        static_var = tf.get_variable(key, dtype=tf.int64)
                        model._dict_append(static_var_dict, key, static_var)
                        zero_assigner = static_var.assign(0)
                        model._dict_append(zero_assigner_dict, key, zero_assigner)

                    for key in ["current_kinetic", "old_total_energy"]:
                        placeholder = tf.placeholder(dds_basetype, name=key)
                        model._dict_append(placeholder_dict, key, placeholder)
                        assigner = static_var_dict[key][i].assign(placeholder)
                        model._dict_append(assigner_dict, key, assigner)

        return static_var_dict, zero_assigner_dict, assigner_dict, placeholder_dict



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
        if "HamiltonianMonteCarlo" in self.FLAGS.sampler:
            return self._sample_Metropolis(return_run_info, return_trajectories, return_averages)
        else:
            return self._sample_LangevinDynamics(return_run_info, return_trajectories, return_averages)

    def _get_test_nodes(self, setup="sample"):
        """ Helper function to create list of nodes for activating sampling or
        training step.

        :param setup: sample or train
        """
        if setup == "sample":
            list_of_nodes = ["sample_step"]
        elif setup == "train":
            list_of_nodes = ["train_step"]
        else:
            logging.critical("Unknown setup parameter in _get_test_nodes().")
            assert(False)
        list_of_nodes.extend(["accuracy", "global_step", "loss"])
        if self.FLAGS.summaries_path is not None:
            test_nodes = [self.summary]*self.FLAGS.number_walkers
        else:
            test_nodes = []
        for item in list_of_nodes:
            test_nodes.append([self.nn[walker_index].get(item) \
                               for walker_index in range(self.FLAGS.number_walkers)])
        return test_nodes

    def _get_all_parameters(self):
        all_weights = []
        all_biases = []
        for walker_index in range(self.FLAGS.number_walkers):
            all_weights.append(self.weights[walker_index].parameters)
            all_biases.append(self.biases[walker_index].parameters)
        return all_weights, all_biases

    def _print_optimizer_parameters(self, feed_dict):
        for walker_index in range(self.FLAGS.number_walkers):
            logging.info("Dependent walker #"+str(walker_index))
            deltat = self.sess.run(self.nn[walker_index].get_list_of_nodes(
                ["learning_rate"]), feed_dict=feed_dict)[0]
            logging.info("GD optimizer parameters, walker #%d: delta t = %lg" % (walker_index, deltat))

    def _print_sampler_parameters(self, feed_dict):
        for walker_index in range(self.FLAGS.number_walkers):
            logging.info("Dependent walker #"+str(walker_index))
            if self.FLAGS.covariance_blending != 0.:
                eta = self.sess.run(self.nn[walker_index].get_list_of_nodes(
                    ["covariance_blending"]), feed_dict=feed_dict)[0]
                logging.info("EQN parameters, walker #%d: eta = %lg" % (walker_index, eta))
            if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                      "GeometricLangevinAlgorithm_1stOrder",
                                      "GeometricLangevinAlgorithm_2ndOrder",
                                      "BAOAB",
                                      "CovarianceControlledAdaptiveLangevinThermostat"]:
                gamma, beta, deltat = self.sess.run(self.nn[walker_index].get_list_of_nodes(
                    ["friction_constant", "inverse_temperature", "step_width"]), feed_dict=feed_dict)
                logging.info("LD Sampler parameters, walker #%d: gamma = %lg, beta = %lg, delta t = %lg" %
                      (walker_index, gamma, beta, deltat))
            elif "HamiltonianMonteCarlo" in self.FLAGS.sampler:
                current_step, num_mc_steps, hd_steps, deltat = self.sess.run(self.nn[walker_index].get_list_of_nodes(
                    ["current_step", "next_eval_step", "hamiltonian_dynamics_steps", "step_width"]), feed_dict=feed_dict)
                logging.info("MC Sampler parameters, walker #%d: current_step = %lg, num_mc_steps = %lg, HD_steps = %lg, delta t = %lg" %
                      (walker_index, current_step, num_mc_steps, hd_steps, deltat))
            else:
                raise NotImplementedError("The sampler method %s is unknown" % (self.FLAGS.sampler))

    def _prepare_summaries(self):
        summary_writer = None
        if self.FLAGS.summaries_path is not None:
            summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_path, self.sess.graph)
        return summary_writer

    def _write_summaries(self, summary_writer, summary, current_step):
        if self.FLAGS.summaries_path is not None:
            summary_writer.add_run_metadata(summary[1], 'step%d' % current_step)
            summary_writer.add_summary(summary[0], current_step)

    def _zero_state_variables(self, method):
        if method in ["GradientDescent",
                      "StochasticGradientLangevinDynamics",
                      "GeometricLangevinAlgorithm_1stOrder",
                      "GeometricLangevinAlgorithm_2ndOrder",
                      "HamiltonianMonteCarlo_1stOrder",
                      "HamiltonianMonteCarlo_2ndOrder",
                      "BAOAB",
                      "CovarianceControlledAdaptiveLangevinThermostat"]:
            check_kinetic, check_inertia, check_momenta, check_gradients, check_virials, check_noise = \
                self.sess.run([
                    self.zero_assigner["kinetic_energy"],
                    self.zero_assigner["inertia"],
                    self.zero_assigner["momenta"],
                    self.zero_assigner["gradients"],
                    self.zero_assigner["virials"],
                    self.zero_assigner["noise"]])
            for walker_index in range(self.FLAGS.number_walkers):
                assert (abs(check_kinetic[walker_index]) < 1e-10)
                assert (abs(check_inertia[walker_index]) < 1e-10)
                assert (abs(check_momenta[walker_index]) < 1e-10)
                assert (abs(check_gradients[walker_index]) < 1e-10)
                assert (abs(check_virials[walker_index]) < 1e-10)
                assert (abs(check_noise[walker_index]) < 1e-10)

    def _get_parameters(self, return_trajectories, all_weights, all_biases):
        weights_eval, biases_eval = None, None
        if self.config_map["do_write_trajectory_file"] or return_trajectories:
            weights_eval, biases_eval = self.sess.run([all_weights, all_biases])
            # [logging.info(str(item)) for item in weights_eval]
            # [logging.info(str(item)) for item in biases_eval]
        return weights_eval, biases_eval

    def _perform_step(self, test_nodes, feed_dict):
        summary = None
        run_metadata = None
        if self.FLAGS.summaries_path is not None:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            results = self.sess.run(test_nodes,
                                    feed_dict=feed_dict,
                                    options=run_options,
                                    run_metadata=run_metadata)
            summary, acc, global_step, loss_eval = \
                results[0], results[2], results[3], results[4]

        else:
            results = self.sess.run(test_nodes,
                                    feed_dict=feed_dict)
            acc, global_step, loss_eval = \
                results[1], results[2], results[3]

        return [summary, run_metadata], acc, global_step, loss_eval

    def _get_elapsed_time_per_nth_step(self, current_step):
        current_time = time.time()
        time_elapsed_per_nth_step = current_time - self.last_time
        if current_step > 1:
            self.elapsed_time += time_elapsed_per_nth_step
            estimated_time_left = (self.FLAGS.max_steps - current_step) * self.elapsed_time / (current_step - 1)
        else:
            estimated_time_left = 0.
        logging.debug("Output at step #" + str(current_step) \
                      + ", est. remaining time is " + str(estimated_time_left) + " seconds.")
        self.last_time = current_time
        return time_elapsed_per_nth_step

    def _decide_collapse_walkers(self, current_step):
        # collapse walkers' positions onto first walker if desired and after having
        # recomputed the covariance matrix
        if (self.FLAGS.number_walkers > 1) and (self.FLAGS.collapse_walkers) and \
                (current_step % self.FLAGS.covariance_after_steps == 0) and \
                (current_step != 0):
            #print("COLLAPSING " + str(self.FLAGS.collapse_walkers))
            # get walker 0's position
            weights_eval, biases_eval = self.sess.run([
                self.weights[0].parameters, self.biases[0].parameters])

            # reset positions of walker 1 till end to that of walker 0
            # assign all in a single session run to allow parallelization
            collapse_feed_dict = {}
            assigns = []
            for walker_index in range(1, self.FLAGS.number_walkers):
                # directly connecting the flat parameters tensor with the respective
                # other walker's parameters' placeholder does not seem to work, i.e.
                # replacing weights_eval -> self.weights[0].parameters
                assert (len(self.weights[0].parameters) == len(self.weights[walker_index].placeholders))
                for weight, weight_placeholder in zip(weights_eval,
                                                      self.weights[walker_index].placeholders):
                    collapse_feed_dict[weight_placeholder] = weight
                assert (len(self.biases[0].parameters) == len(self.biases[walker_index].placeholders))
                for bias, bias_placeholder in zip(biases_eval,
                                                  self.biases[walker_index].placeholders):
                    collapse_feed_dict[bias_placeholder] = bias
                assigns.append(self.weights[walker_index].assign_all_t)
                assigns.append(self.biases[walker_index].assign_all_t)
            # evaluate and assign all at once
            self.sess.run(assigns, feed_dict=collapse_feed_dict)

    def _set_HMC_placeholders(self, HMC_placeholder_nodes, current_step, step_widths, HD_steps, HMC_steps, feed_dict):
        if "HamiltonianMonteCarlo" in self.FLAGS.sampler:
            for walker_index in range(self.FLAGS.number_walkers):
                feed_dict.update({
                    HMC_placeholder_nodes[walker_index]["step_width"]: step_widths[walker_index],
                    HMC_placeholder_nodes[walker_index]["next_eval_step"]: HMC_steps[walker_index],
                    HMC_placeholder_nodes[walker_index]["hamiltonian_dynamics_steps"]: HD_steps[walker_index]
                })
                feed_dict.update({
                    HMC_placeholder_nodes[walker_index]["current_step"]: current_step
                })
        return feed_dict

    def _set_HMC_next_eval_step(self, current_step, step_widths, HD_steps, HMC_steps):
        if "HamiltonianMonteCarlo" in self.FLAGS.sampler:
            for walker_index in range(self.FLAGS.number_walkers):
                if current_step > HMC_steps[walker_index]:
                    # pick next evaluation step with a little random variation
                    step_widths[walker_index] = \
                        np.random.uniform(low=0.7, high=1.3) * self.FLAGS.step_width
                    logging.debug("Next step width of #"+str(walker_index) \
                                  +" is " + str(step_widths[walker_index]))

                    # pick next evaluation step with a little random variation
                    HD_steps[walker_index] = \
                        max(1, round((0.9 + np.random.uniform(low=0., high=0.2)) \
                                     * self.FLAGS.hamiltonian_dynamics_time / self.FLAGS.step_width))
                    if self.FLAGS.sampler == "HamiltonianMonteCarlo_1stOrder":
                        # one extra step for the criterion evaluation
                        HMC_steps[walker_index] += 1 + HD_steps[walker_index]
                    elif self.FLAGS.sampler == "HamiltonianMonteCarlo_2ndOrder":
                        # with Leapfrog integration we need an additional step
                        # for the last "B" step of BAB due to cyclic permutation
                        # to BBA.
                        HMC_steps[walker_index] += 2 + HD_steps[walker_index]
                    else:
                        raise NotImplementedError("The HMC sampler method %S is unknown" % (self.FLAGS.sampler))
                    logging.debug("Next amount of HD steps is " + str(HD_steps)
                                  +", evaluation of HMC criterion at step " + str(HMC_steps))
        else:
            for walker_index in range(self.FLAGS.number_walkers):
                HMC_steps[walker_index] = current_step
        return HD_steps, HMC_steps

    def _set_HMC_eval_variables(self, current_step, HMC_steps, values):
        if "HamiltonianMonteCarlo" in self.FLAGS.sampler:
            # set current kinetic as it is accumulated outside of tensorflow
            kin_eval = self.sess.run(self.static_vars["kinetic_energy"])
            set_dict = {}
            for walker_index in range(self.FLAGS.number_walkers):
                set_dict[ self.placeholders["current_kinetic"][walker_index] ] = \
                    kin_eval[walker_index]
            self.sess.run(self.assigner["current_kinetic"], feed_dict=set_dict)

            # possibly reset some old energy values for acceptance criterion if
            # an acceptance evaluation has just occured
            #
            # Note that we always evaluate the criterion in step 0 and make sure that
            # the we always accept (by having old total and current energy coincide)
            # such that a valid old parameter set is stored to which we may restore
            # if the next evaluation rejects.
            do_evaluate = False
            for walker_index in range(self.FLAGS.number_walkers):
                if current_step > HMC_steps[walker_index] or current_step == 0:
                    # at least one walker requires a loss calculation
                    do_evaluate = True
            if do_evaluate or self.FLAGS.verbose > 1:
                HMC_set_total_energy = []
                energy_placeholders = {}
                for walker_index in range(self.FLAGS.number_walkers):
                    if current_step > HMC_steps[walker_index] or current_step == 0:
                        HMC_set_total_energy.extend([self.assigner["old_total_energy"][walker_index]])
                        energy_placeholders[self.placeholders["old_total_energy"][walker_index]] = \
                            values.loss[walker_index]+values.kinetic_energy[walker_index]
                        logging.debug("Resetting total energy for walker #"+str(walker_index))
                if len(HMC_set_total_energy) > 0:
                    total_eval = self.sess.run(HMC_set_total_energy, feed_dict=energy_placeholders)
                    logging.debug("New total energies are "+str(total_eval))

    def _prepare_HMC_nodes(self):
        if "HamiltonianMonteCarlo" in self.FLAGS.sampler:
            # zero rejection rate before sampling start
            check_accepted, check_rejected = self.sess.run([
                self.zero_assigner["accepted"], self.zero_assigner["rejected"]])
            for walker_index in range(self.FLAGS.number_walkers):
                assert(check_accepted[walker_index] == 0)
                assert(check_rejected[walker_index] == 0)

    def _get_trajectory_header(self):
        if self.directions is not None:
            print(self.directions.shape)
            header = get_trajectory_header(
                self.directions.shape[0],
                0)
        else:
            header = get_trajectory_header(
                self.weights[0].get_total_dof(),
                self.biases[0].get_total_dof())
        return header

    def _sample_Metropolis(self, return_run_info=False, return_trajectories=False, return_averages=False):

        self.init_files("sample")

        HMC_placeholder_nodes = [self.nn[walker_index].get_dict_of_nodes(
            ["current_step", "next_eval_step", "step_width", "hamiltonian_dynamics_steps"])
            for walker_index in range(self.FLAGS.number_walkers)]

        test_nodes = self._get_test_nodes(setup="sample")
        EQN_nodes =[self.nn[walker_index].get("EQN_step") \
                           for walker_index in range(self.FLAGS.number_walkers)]
        all_weights, all_biases = self._get_all_parameters()

        averages = AveragesAccumulator(return_averages, self.FLAGS.sampler,
                                       self.config_map,
                                       self.averages_writer,
                                       header=self.get_averages_header(setup="sample"),
                                       max_steps=self.FLAGS.max_steps,
                                       every_nth=self.FLAGS.every_nth,
                                       inverse_temperature=self.FLAGS.inverse_temperature,
                                       burn_in_steps=self.FLAGS.burn_in_steps,
                                       number_walkers=self.FLAGS.number_walkers)
        run_info = RuninfoAccumulator(return_run_info, self.FLAGS.sampler,
                                      self.config_map,
                                      self.run_writer,
                                      header=self.get_sample_header(),
                                      max_steps=self.FLAGS.max_steps,
                                      every_nth=self.FLAGS.every_nth,
                                      number_walkers=self.FLAGS.number_walkers)
        trajectory = TrajectoryAccumulator(return_trajectories, self.FLAGS.sampler,
                                           self.config_map,
                                           self.trajectory_writer,
                                           header=self._get_trajectory_header(),
                                           max_steps=self.FLAGS.max_steps,
                                           every_nth=self.FLAGS.every_nth,
                                           number_walkers=self.FLAGS.number_walkers,
                                           directions=self.directions)
        accumulated_values = AccumulatedValues()

        # place in feed dict: We have to supply all placeholders (regardless of
        # which the employed sampler actually requires) because of the evaluated
        # summary! All of the placeholder nodes are also summary nodes.
        feed_dict = {}
        for walker_index in range(self.FLAGS.number_walkers):
            feed_dict.update(self._create_default_feed_dict_with_constants(walker_index))

        # zero extra nodes for HMC
        self._prepare_HMC_nodes()

        # check that sampler's parameters are actually used
        self._print_sampler_parameters(feed_dict)

        # prepare summaries for TensorBoard
        summary_writer = self._prepare_summaries()

        # prepare some loop variables
        logging.info("Starting to sample")
        logging.info_intervals = max(1, int(self.FLAGS.max_steps / 100))
        self.last_time = time.time()
        self.elapsed_time = 0.
        HD_steps = [-2]*self.FLAGS.number_walkers        # number of hamiltonian dynamics steps
        HMC_steps = [0]*self.FLAGS.number_walkers       # next step where to evaluate criterion
        HMC_old_steps = [0]*self.FLAGS.number_walkers   # last step where criterion was evaluated
        # we need to randomly vary the step widths to avoid (quasi-)periodicities
        step_widths = [self.FLAGS.step_width]*self.FLAGS.number_walkers
        self._set_HMC_placeholders(HMC_placeholder_nodes,
                                   1, step_widths, HD_steps, HMC_steps, feed_dict)
        if tqdm_present and self.FLAGS.progress:
            step_range = tqdm(range(self.FLAGS.max_steps))
        else:
            step_range = range(self.FLAGS.max_steps)

        # backup gradients and virials of each initial state to avoid recalculation
        initial_state_gradients = None
        initial_state_virials = None
        initial_state_inertia = None
        initial_state_momenta = None

        last_rejected = self.sess.run(self.static_vars["rejected"])   # temporary to see whether last evaluation was a rejection
        for current_step in step_range:
            # get next batch of data
            features, labels = self.input_pipeline.next_batch(self.sess)
            # logging.debug("batch is x: "+str(features[:])+", y: "+str(labels[:]))

            # update feed_dict for this step
            feed_dict.update({
                self.xinput: features,
                self.true_labels: labels
            })

            # get energies for acceptance evaluation in very first step
            if current_step == 0:
                accumulated_values.loss, accumulated_values.kinetic_energy = \
                    self.sess.run([test_nodes[3], self.static_vars["kinetic_energy"]], feed_dict=feed_dict)

            # set global variable used in HMC sampler for criterion to initial loss
            self._set_HMC_eval_variables(current_step, HMC_steps, accumulated_values)

            # zero kinetic energy and other variables
            self._zero_state_variables(self.FLAGS.sampler)

            # tell accumulators about next evaluation step (delayed by one)
            run_info.inform_next_eval_step(HMC_steps, accumulated_values.rejected)
            trajectory.inform_next_eval_step(HMC_steps, accumulated_values.rejected)
            averages.inform_next_eval_step(HMC_steps, accumulated_values.rejected)

            # set next criterion evaluation step
            # needs to be after `_set_HMC_eval_variables()`
            # needs to be before `_perform_step()`
            HMC_old_steps[:] = HMC_steps
            HD_steps, HMC_steps = self._set_HMC_next_eval_step(
                current_step, step_widths, HD_steps, HMC_steps)
            feed_dict = self._set_HMC_placeholders(HMC_placeholder_nodes,
                                                   current_step, step_widths,
                                                   HD_steps, HMC_steps, feed_dict)

            # get the weights and biases as otherwise the loss won't match
            # tf first computes loss, then gradient, then performs variable update
            # hence after the sample step, we would have updated variables but old loss
            if current_step % self.FLAGS.every_nth == 0:
                accumulated_values.weights, accumulated_values.biases = \
                    self._get_parameters(return_trajectories, all_weights, all_biases)

            # perform the EQN update step
            if self.FLAGS.covariance_blending != 0. and \
                    current_step % self.FLAGS.covariance_after_steps == 0:
                self.sess.run(EQN_nodes, feed_dict=feed_dict)

            # perform the sampling step
            step_success = False
            blending_key = [self.nn[walker_index].placeholder_nodes["covariance_blending"]
                            for walker_index in range(self.FLAGS.number_walkers)]
            old_eta = [feed_dict[blending_key[walker_index]] for walker_index in range(self.FLAGS.number_walkers)]
            while not step_success:
                for walker_index in range(self.FLAGS.number_walkers):
                    if feed_dict[blending_key[walker_index]] > 0. \
                            and feed_dict[blending_key[walker_index]] < 1e-12:
                        logging.warning("Possible NaNs or Infs in covariance matrix, setting eta to 0 temporarily.")
                        feed_dict[blending_key[walker_index]] = 0.
                try:
                    summary, accumulated_values.accuracy, accumulated_values.global_step, accumulated_values.loss = \
                        self._perform_step(test_nodes, feed_dict)
                    step_success = True
                except tf.errors.InvalidArgumentError as err:
                    # Cholesky failed, try again with smaller eta
                    for walker_index in range(self.FLAGS.number_walkers):
                        feed_dict[blending_key[walker_index]] = feed_dict[blending_key[walker_index]]/2.
                    logging.warning(str(err.op) + " FAILED at step %d, using %lg as eta." \
                                    % (current_step, feed_dict[blending_key[0]]))
            for walker_index in range(self.FLAGS.number_walkers):
                feed_dict[blending_key[walker_index]] = old_eta[walker_index]

            # get updated state variables
            accumulated_values.evaluate(self.sess, self.FLAGS.sampler, self.static_vars)

            def print_energies():
                if self.FLAGS.verbose > 1:
                    for walker_index in range(self.FLAGS.number_walkers):
                        loss_subtext = "n" if accumulated_values.rejected[walker_index] != last_rejected[walker_index] else "n-1"
                        kinetic_subtext = "n-1" if "HamiltonianMonteCarlo_2ndOrder" in self.FLAGS.sampler \
                                and current_step != HMC_old_steps[walker_index] else "n"
                        logging.debug("walker #%d, #%d: L(x_{%s})=%lg, total is %lg, T(p_{%s})=%lg, sum is %lg" \
                                  % (walker_index, current_step, loss_subtext,
                                     # to emphasize updated loss
                                     accumulated_values.loss[walker_index],
                                     accumulated_values.old_total_energy[walker_index],
                                     kinetic_subtext, # for HMC_2nd
                                     accumulated_values.kinetic_energy[walker_index],
                                     accumulated_values.loss[walker_index] + accumulated_values.kinetic_energy[
                                         walker_index]))

            # give output on debug mode
            print_energies()

            # if last step was rejection, re-evaluate loss and weights as state changed
            if accumulated_values.rejected != last_rejected:
                # recalculate loss and get kinetic energy
                accumulated_values.loss, accumulated_values.kinetic_energy = \
                    self.sess.run([test_nodes[3], self.static_vars["kinetic_energy"]], feed_dict=feed_dict)
                # get restored weights and biases
                accumulated_values.weights, accumulated_values.biases = \
                    self._get_parameters(return_trajectories, all_weights, all_biases)
                logging.info("Last state REJECTed.")
                print_energies()

            # reset gradients and virials to initial state's if rejected
            for walker_index in range(self.FLAGS.number_walkers):
                if current_step == HMC_old_steps[walker_index]:
                    # restore gradients and virials
                    if accumulated_values.rejected[walker_index] != last_rejected[walker_index]:
                        accumulated_values.gradients[walker_index] = initial_state_gradients
                        accumulated_values.virials[walker_index] = initial_state_virials
                        accumulated_values.inertia[walker_index] = initial_state_inertia
                        accumulated_values.momenta[walker_index] = initial_state_momenta
                    else:
                        initial_state_gradients = accumulated_values.gradients[walker_index]
                        initial_state_virials = accumulated_values.virials[walker_index]
                        initial_state_inertia = accumulated_values.inertia[walker_index]
                        initial_state_momenta = accumulated_values.momenta[walker_index]
                    # accumulate averages and other information
                    if current_step >= self.FLAGS.burn_in_steps:
                        averages.accumulate_each_step(current_step, walker_index, accumulated_values)

            # write summaries for tensorboard
            self._write_summaries(summary_writer, summary, current_step)

            if current_step % self.FLAGS.every_nth == 0:
                accumulated_values.time_elapsed_per_nth_step = self._get_elapsed_time_per_nth_step(current_step)

            for walker_index in range(self.FLAGS.number_walkers):
                run_info.accumulate_nth_step(current_step, walker_index, accumulated_values)
                trajectory.accumulate_nth_step(current_step, walker_index, accumulated_values)
                averages.accumulate_nth_step(current_step, walker_index, accumulated_values)

            # update temporary rejected
            last_rejected = accumulated_values.rejected

            #if (i % logging.info_intervals) == 0:
                #logging.debug('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                #logging.debug('Loss at step %s: %s' % (i, loss_eval))

            self._decide_collapse_walkers(current_step)

        logging.info("SAMPLED.")

        # close summaries file
        if self.FLAGS.summaries_path is not None:
            summary_writer.close()

        self.finish_files()

        return run_info.run_info, trajectory.trajectory, averages.averages

    def _sample_LangevinDynamics(self, return_run_info=False, return_trajectories=False, return_averages=False):

        self.init_files("sample")

        placeholder_nodes = []
        for walker_index in range(self.FLAGS.number_walkers):
            placeholder_nodes.append(self.nn[walker_index].get_dict_of_nodes(["current_step"]))
        test_nodes = self._get_test_nodes(setup="sample")
        EQN_nodes =[self.nn[walker_index].get("EQN_step") \
                           for walker_index in range(self.FLAGS.number_walkers)]
        all_weights, all_biases = self._get_all_parameters()

        averages = AveragesAccumulator(return_averages, self.FLAGS.sampler,
                                       self.config_map,
                                       self.averages_writer,
                                       header=self.get_averages_header(setup="sample"),
                                       max_steps=self.FLAGS.max_steps,
                                       every_nth=self.FLAGS.every_nth,
                                       inverse_temperature=self.FLAGS.inverse_temperature,
                                       burn_in_steps=self.FLAGS.burn_in_steps,
                                       number_walkers=self.FLAGS.number_walkers)
        run_info = RuninfoAccumulator(return_run_info, self.FLAGS.sampler,
                                      self.config_map,
                                      self.run_writer,
                                      header=self.get_sample_header(),
                                      max_steps=self.FLAGS.max_steps,
                                      every_nth=self.FLAGS.every_nth,
                                      number_walkers=self.FLAGS.number_walkers)
        trajectory = TrajectoryAccumulator(return_trajectories, self.FLAGS.sampler,
                                           self.config_map,
                                           self.trajectory_writer,
                                           header=self._get_trajectory_header(),
                                           max_steps=self.FLAGS.max_steps,
                                           every_nth=self.FLAGS.every_nth,
                                           number_walkers=self.FLAGS.number_walkers,
                                           directions=self.directions)
        accumulated_values = AccumulatedValues()

        # place in feed dict: We have to supply all placeholders (regardless of
        # which the employed sampler actually requires) because of the evaluated
        # summary! All of the placeholder nodes are also summary nodes.
        feed_dict = {}
        for walker_index in range(self.FLAGS.number_walkers):
            feed_dict.update(self._create_default_feed_dict_with_constants(walker_index))

        # check that sampler's parameters are actually used
        self._print_sampler_parameters(feed_dict)

        # prepare summaries for TensorBoard
        summary_writer = self._prepare_summaries()

        # prepare some loop variables
        logging.info("Starting to sample")
        logging.info_intervals = max(1, int(self.FLAGS.max_steps / 100))
        self.last_time = time.time()
        self.elapsed_time = 0.
        if tqdm_present and self.FLAGS.progress:
            step_range = tqdm(range(self.FLAGS.max_steps))
        else:
            step_range = range(self.FLAGS.max_steps)

        for current_step in step_range:
            # get next batch of data
            features, labels = self.input_pipeline.next_batch(self.sess)
            # logging.debug("batch is x: "+str(features[:])+", y: "+str(labels[:]))

            # update feed_dict for this step
            feed_dict.update({
                self.xinput: features,
                self.true_labels: labels
            })
            for walker_index in range(self.FLAGS.number_walkers):
                feed_dict.update({
                    placeholder_nodes[walker_index]["current_step"]: current_step
                })

            # zero kinetic energy and other variables
            self._zero_state_variables(self.FLAGS.sampler)

            # get the weights and biases as otherwise the loss won't match
            # tf first computes loss, then gradient, then performs variable update
            # hence after the sample step, we would have updated variables but old loss
            if current_step % self.FLAGS.every_nth == 0:
                accumulated_values.weights, accumulated_values.biases = \
                    self._get_parameters(return_trajectories, all_weights, all_biases)

            # perform the EQN update step
            if self.FLAGS.covariance_blending != 0. and \
                    current_step % self.FLAGS.covariance_after_steps == 0:
                self.sess.run(EQN_nodes, feed_dict=feed_dict)

            # perform the sampling step
            step_success = False
            blending_key = [self.nn[walker_index].placeholder_nodes["covariance_blending"]
                            for walker_index in range(self.FLAGS.number_walkers)]
            old_eta = [feed_dict[blending_key[walker_index]] for walker_index in range(self.FLAGS.number_walkers)]
            while not step_success:
                for walker_index in range(self.FLAGS.number_walkers):
                    if feed_dict[blending_key[walker_index]] > 0. \
                            and feed_dict[blending_key[walker_index]] < 1e-12:
                        logging.warning("Possible NaNs or Infs in covariance matrix, setting eta to 0 temporarily.")
                        feed_dict[blending_key[walker_index]] = 0.
                try:
                    summary, accumulated_values.accuracy, accumulated_values.global_step, accumulated_values.loss = \
                        self._perform_step(test_nodes, feed_dict)
                    step_success = True
                except tf.errors.InvalidArgumentError as err:
                    # Cholesky failed, try again with smaller eta
                    for walker_index in range(self.FLAGS.number_walkers):
                        feed_dict[blending_key[walker_index]] = feed_dict[blending_key[walker_index]]/2.
                    logging.warning(str(err.op) + " FAILED at step %d, using %lg as eta." \
                                    % (current_step, feed_dict[blending_key[0]]))
            for walker_index in range(self.FLAGS.number_walkers):
                feed_dict[blending_key[walker_index]] = old_eta[walker_index]

            # get updated state variables
            accumulated_values.evaluate(self.sess, self.FLAGS.sampler, self.static_vars)

            # write summaries for tensorboard
            self._write_summaries(summary_writer, summary, current_step)

            # accumulate averages
            if current_step >= self.FLAGS.burn_in_steps:
                for walker_index in range(self.FLAGS.number_walkers):
                    averages.accumulate_each_step(current_step, walker_index, accumulated_values)

            if current_step % self.FLAGS.every_nth == 0:
                accumulated_values.time_elapsed_per_nth_step = self._get_elapsed_time_per_nth_step(current_step)

            for walker_index in range(self.FLAGS.number_walkers):
                run_info.accumulate_nth_step(current_step, walker_index, accumulated_values)
                trajectory.accumulate_nth_step(current_step, walker_index, accumulated_values)
                averages.accumulate_nth_step(current_step, walker_index, accumulated_values)

            #if (i % logging.info_intervals) == 0:
                #logging.debug('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                #logging.debug('Loss at step %s: %s' % (i, loss_eval))

            self._decide_collapse_walkers(current_step)

        logging.info("SAMPLED.")

        # close summaries file
        if self.FLAGS.summaries_path is not None:
            summary_writer.close()

        self.finish_files()

        return run_info.run_info, trajectory.trajectory, averages.averages

    def _create_default_feed_dict_with_constants(self, walker_index=0):
        """ Construct an initial feed dict from all constant parameters
        such as step width, ...

        Here, we check whether the respective placeholder node is contained
        in the neural network and only in that case add the value to the
        feed_dict.

        Basically, we connect entries in the "FLAGS" structure that is parsed
        from cmd-line or created through `setup_parameters()` with the slew of
        placeholders in tensorflow's neural network.

        :param walker_index: index of walker whose placeholders to feed
        :return: feed_dict with constant parameters
        """

        # add sampler options only when they are present in parameter struct
        param_dict = {}
        for key in ["covariance_blending",
                    "friction_constant", "inverse_temperature",
                    "learning_rate", "sigma", "sigmaA", "step_width"]:
            try:
                param_dict[key] = getattr(self.FLAGS, key)
            except AttributeError:
                pass
        # special case because key and attribute's name differ
        try:
            param_dict["next_eval_step"] = 0
            param_dict["hamiltonian_dynamics_steps"] = 0
        except AttributeError:
            pass

        # add other options that are present in any case
        param_dict.update({
            "current_step": 0,
            "keep_probability": self.FLAGS.dropout if self.FLAGS.dropout is not None else 0.})

        # for each parameter check for placeholder and add to dict on its presence
        default_feed_dict = {}
        for key in param_dict.keys():
            if key in self.nn[walker_index].placeholder_nodes.keys():
                default_feed_dict.update({
                    self.nn[walker_index].placeholder_nodes[key]: param_dict[key]})

        return default_feed_dict


    def train(self, walker_index=0, return_run_info = False, return_trajectories = False, return_averages=False):
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
        self.init_files("train")

        assert( walker_index < self.FLAGS.number_walkers)

        test_nodes = self._get_test_nodes(setup="train")
        EQN_nodes =[self.nn[walker_index].get("EQN_step") \
                           for walker_index in range(self.FLAGS.number_walkers)]
        all_weights, all_biases = self._get_all_parameters()

        averages = AveragesAccumulator(return_averages, self.FLAGS.optimizer,
                                       self.config_map,
                                       self.averages_writer,
                                       header=self.get_averages_header(setup="train"),
                                       max_steps=self.FLAGS.max_steps,
                                       every_nth=self.FLAGS.every_nth,
                                       burn_in_steps=self.FLAGS.burn_in_steps,
                                       number_walkers=self.FLAGS.number_walkers)
        run_info = RuninfoAccumulator(return_run_info, self.FLAGS.optimizer,
                                      self.config_map,
                                      self.run_writer,
                                      header=self.get_train_header(),
                                      max_steps=self.FLAGS.max_steps,
                                      every_nth=self.FLAGS.every_nth,
                                      number_walkers=self.FLAGS.number_walkers)
        trajectory = TrajectoryAccumulator(return_trajectories, self.FLAGS.optimizer,
                                           self.config_map,
                                           self.trajectory_writer,
                                           header=self._get_trajectory_header(),
                                           max_steps=self.FLAGS.max_steps,
                                           every_nth=self.FLAGS.every_nth,
                                           number_walkers=self.FLAGS.number_walkers,
                                           directions=self.directions)
        accumulated_values = AccumulatedValues()

        # place in feed dict: We have to supply all placeholders (regardless of
        # which the employed optimizer actually requires) because of the evaluated
        # summary! All of the placeholder nodes are also summary nodes.
        feed_dict = {}
        for walker_index in range(self.FLAGS.number_walkers):
            feed_dict.update(self._create_default_feed_dict_with_constants(walker_index))

        # check that optimizers's parameters are actually used
        self._print_optimizer_parameters(feed_dict)

        # prepare summaries for TensorBoard
        summary_writer = self._prepare_summaries()

        # prepare some loop variables
        logging.info("Starting to train")
        logging.info_intervals = max(1, int(self.FLAGS.max_steps / 100))
        self.last_time = time.time()
        self.elapsed_time = 0
        if tqdm_present and self.FLAGS.progress:
            step_range = tqdm(range(self.FLAGS.max_steps))
        else:
            step_range = range(self.FLAGS.max_steps)

        for current_step in step_range:
            # get next batch of data
            features, labels = self.input_pipeline.next_batch(self.sess)
            # logging.debug("batch is x: "+str(features[:])+", y: "+str(labels[:]))

            # update feed_dict for this step
            feed_dict.update({
                self.xinput: features,
                self.true_labels: labels
            })

            # zero kinetic energy and other variables
            self._zero_state_variables(self.FLAGS.optimizer)

            # get the weights and biases as otherwise the loss won't match
            # tf first computes loss, then gradient, then performs variable update
            # hence after the sample step, we would have updated variables but old loss
            if current_step % self.FLAGS.every_nth == 0:
                accumulated_values.weights, accumulated_values.biases = \
                    self._get_parameters(return_trajectories, all_weights, all_biases)

            ## training is not compatible (yet) with performing the EQN update step
            #if self.FLAGS.covariance_blending != 0. and \
            #        current_step % self.FLAGS.covariance_after_steps == 0:
            #    self.sess.run(EQN_nodes, feed_dict=feed_dict)

            # perform the sampling step
            summary, accumulated_values.accuracy, accumulated_values.global_step, accumulated_values.loss = \
                self._perform_step(test_nodes, feed_dict)

            # get updated state variables
            accumulated_values.evaluate(self.sess, self.FLAGS.optimizer, self.static_vars)

            # write summaries for tensorboard
            self._write_summaries(summary_writer, summary, current_step)

            # accumulate averages
            if current_step >= self.FLAGS.burn_in_steps:
                for walker_index in range(self.FLAGS.number_walkers):
                    averages.accumulate_each_step(current_step, walker_index, accumulated_values)

            if current_step % self.FLAGS.every_nth == 0:
                accumulated_values.time_elapsed_per_nth_step = self._get_elapsed_time_per_nth_step(current_step)

            for walker_index in range(self.FLAGS.number_walkers):
                run_info.accumulate_nth_step(current_step, walker_index, accumulated_values)
                trajectory.accumulate_nth_step(current_step, walker_index, accumulated_values)
                averages.accumulate_nth_step(current_step, walker_index, accumulated_values)

            #if (i % logging.info_intervals) == 0:
                #logging.debug('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                #logging.debug('Loss at step %s: %s' % (i, loss_eval))

            self._decide_collapse_walkers(current_step)

        logging.info("TRAINED down to loss %s and accuracy %s." %
                     (accumulated_values.loss[0], accumulated_values.accuracy[0]))

        # close summaries file
        if self.FLAGS.summaries_path is not None:
            summary_writer.close()

        self.finish_files()

        # get rid of possile arrays (because of multiple walkers) in return arrays
        ret_vals = [None, None, None]
        if run_info.run_info is not None:
            ret_vals[0] = run_info.run_info[0]
        if trajectory.trajectory is not None:
            ret_vals[1] = trajectory.trajectory[0]
        if averages.averages is not None:
            ret_vals[2] = averages.averages[0]
        return ret_vals

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


    def close_files(self):
        """ Closes the output files if they have been opened.
        """
        if self.config_map["do_write_averages_file"]:
            assert self.config_map["averages_file"] is not None
            self.config_map["averages_file"].close()
            self.config_map["averages_file"] = None
            self.averages_writer = None
        if self.config_map["do_write_run_file"]:
            assert self.config_map["csv_file"] is not None
            self.config_map["csv_file"].close()
            self.config_map["csv_file"] = None
            self.run_writer = None
        if self.config_map["do_write_trajectory_file"]:
            assert self.config_map["trajectory_file"] is not None
            self.config_map["trajectory_file"].close()
            self.config_map["trajectory_file"] = None
            self.trajectory_writer = None

    def save_model(self, filename):
        """ Saves the current neural network model to a set of files,
        whose prefix is given by filename.

        :param filename: prefix of set of model files
        :return: path where model was saved
        """
        return self.saver.save(self.sess, filename)

    def finish_files(self):
        """ Closes all open files and saves the model if desired
        """
        self.close_files()

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
            sample_step_placeholder = self.step_placeholder[walker_index]
            feed_dict = {sample_step_placeholder: step}
            set_step = self.sess.run(self.global_step_assign_t[walker_index], feed_dict=feed_dict)
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
