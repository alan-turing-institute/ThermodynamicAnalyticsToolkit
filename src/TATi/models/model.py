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
        self.fixed_variables = {}

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

        # mark step assign op as to be created
        self.step_placeholder = None
        self.global_step_assign_t = None

        # mark writer as to be created
        self.averages_writer = None
        self.run_writer = None
        self.trajectory_writer = None

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
        static_vars_float = ["old_loss", "old_kinetic", "kinetic_energy", "total_energy", "momenta", \
                             "gradients", "virials", "noise"]
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
            scope_name = 'walker'+str(i+1)
            for var in _collection:
                if scope_name in var.name:
                    split_collection[-1].append(var)
        return split_collection

    def init_network(self, filename = None, setup = None,
                     add_vectorized_gradients = False):
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
        assert( self.input_dimension is not None )

        # create input layer
        if self.xinput is None or self.x is None:
            input_columns = get_list_from_string(self.FLAGS.input_columns)
            self.xinput, self.x = create_input_layer(self.input_dimension, input_columns)

        fixed_variables = []
        if self.nn is None:
            self.nn = []
            self.loss = []
            self.trainables = []
            if self.FLAGS.do_hessians or add_vectorized_gradients:
                self.gradients = []
                if self.FLAGS.do_hessians:
                    self.hessians = []
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

            # construct (vectorized) gradient nodes and hessians
            for i in range(self.FLAGS.number_walkers):
                with tf.name_scope('walker' + str(i + 1)):
                    if self.FLAGS.do_hessians or add_vectorized_gradients:
                        # create node for gradient and hessian computation only if specifically
                        # requested as the creation along is costly (apart from the expensive part
                        # of evaluating the nodes eventually). This would otherwise slow down
                        # startup quite a bit even when hessians are not evaluated.
                        #print("GRADIENTS")
                        vectorized_gradients = []
                        trainables = tf.get_collection_ref(self.trainables[i])
                        for tensor in trainables:
                            grad = tf.gradients(self.loss, tensor)
                            #print(grad)
                            vectorized_gradients.append(tf.reshape(grad, [-1]))
                        self.gradients.append(tf.reshape(tf.concat(vectorized_gradients, axis=0), [-1]))

                    if self.FLAGS.do_hessians:
                        #print("HESSIAN")
                        total_dofs = 0
                        hessians = []
                        for gradient in vectorized_gradients:
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
        else:
            self.loss = []
            for i in range(self.FLAGS.number_walkers):
                self.loss.append(self.nn[i].get_list_of_nodes(["loss"])[0])

                if self.FLAGS.fix_parameters is not None:
                    names, values = self.split_parameters_as_names_values(self.FLAGS.fix_parameters)
                    fixed_variables.extend(self.fix_parameters(names))
                    logging.info("Excluded the following degrees of freedom: " + str(fixed_variables))

        logging.debug("Remaining global trainable variables: " + str(variables.trainable_variables()))

        # create global variables, one for every walker in its replicated graph
        self.create_resource_variables()
        self.static_vars, self.zero_assigner, self.assigner = \
            self._create_static_variable_dict(self.FLAGS.number_walkers)

        # set number of degrees of freedom
        split_weights = self._split_collection_per_walker(
            tf.get_collection_ref(tf.GraphKeys.WEIGHTS), self.FLAGS.number_walkers)
        split_biases = self._split_collection_per_walker(
            tf.get_collection_ref(tf.GraphKeys.BIASES), self.FLAGS.number_walkers)
        self.number_of_parameters = \
            neuralnet_parameters.get_total_dof_from_list(split_weights[0]) \
            + neuralnet_parameters.get_total_dof_from_list(split_biases[0])
        logging.info("Number of dof per walker: "+str(self.number_of_parameters))

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

        # setup training/sampling
        if setup is not None:
            if "train" in setup:
                self.optimizer = []
                for i in range(self.FLAGS.number_walkers):
                    with tf.variable_scope("var_walker" + str(i + 1)):
                    #with tf.name_scope('gradients_walker'+str(i+1)):
                        self.optimizer.append(self.nn[i].add_train_method(self.loss[i], optimizer_method=self.FLAGS.optimizer, prior=prior))
            if "sample" in setup:
                self.sampler = []
                for i in range(self.FLAGS.number_walkers):
                    if self.FLAGS.seed is not None:
                        walker_seed = self.FLAGS.seed + i
                    else:
                        walker_seed = self.FLAGS.seed
                    self.sampler.append(self.nn[i]._prepare_sampler(
                        self.loss[i], sampling_method=self.FLAGS.sampler,
                        seed=walker_seed, prior=prior,
                        sigma=self.FLAGS.sigma, sigmaA=self.FLAGS.sigmaA,
                        covariance_blending=self.FLAGS.covariance_blending))
                # create gradients
                grads_and_vars = []
                for i in range(self.FLAGS.number_walkers):
                    with tf.name_scope('gradients_walker'+str(i+1)):
                        trainables = tf.get_collection_ref(self.trainables[i])
                        grads_and_vars.append(self.sampler[i].compute_and_check_gradients(
                            self.loss[i], var_list=trainables))

                # combine gradients
                #for i in range(self.FLAGS.number_walkers):
                #    print("walker "+str(i))
                #    var_list = [v for g, v in grads_and_vars[i] if g is not None]
                #    grad_list = [g for g, v in grads_and_vars[i] if g is not None]
                #    for j in range(len(grad_list)):
                #         print(var_list[j])

                # add position update nodes
                for i in range(self.FLAGS.number_walkers):
                    with tf.variable_scope("var_walker" + str(i + 1)):
                        global_step = self.nn[i]._prepare_global_step()
                        train_step = self.sampler[i].apply_gradients(
                            grads_and_vars, i, global_step=global_step,
                            name=self.sampler[i].get_name())
                    self.nn[i].summary_nodes['sample_step'] = train_step

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
        else:
            logging.info("Not adding sample or train method.")

        # setup model saving/recovering
        if self.saver is None:
            self.saver = tf.train.Saver(tf.get_collection_ref(tf.GraphKeys.WEIGHTS) +
                                   tf.get_collection_ref(tf.GraphKeys.BIASES) + \
                                   tf.get_collection_ref("Variables_to_Save"))

        # merge summaries at very end
        self.summary = tf.summary.merge_all()  # Merge all the summaries

        if self.sess is None:
            logging.debug("Using %s, %s threads " % (str(self.FLAGS.intra_ops_threads), str(self.FLAGS.inter_ops_threads)))
            self.sess = tf.Session(
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=self.FLAGS.intra_ops_threads,
                    inter_op_parallelism_threads=self.FLAGS.inter_ops_threads))

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

        ### Now the session object is created, graph must be done here!

        # initialize constants in graph
        NeuralNetwork.init_graph(self.sess)

        # initialize dataset
        #self.input_pipeline.reset(self.sess)

        if self.FLAGS.fix_parameters is not None:
            logging.debug("Assigning the following values to fixed degrees of freedom: "+str(values))
            self.sess.run(fix_parameter_assigns)

        # assign state of model from file if given
        if filename is not None:
            # Tensorflow DOCU says: initializing is not needed when restoring
            # however, global_variables are missing otherwise for storing kinetic, ...
            # tf.reset_default_graph()

            restore_path = filename.replace('.meta', '')
            self.saver.restore(self.sess, restore_path)
            logging.info("Model restored from file: %s" % restore_path)

        # assign parameters of NN from step in given file
        if self.FLAGS.parse_parameters_file is not None \
                and (self.FLAGS.parse_steps is not None and (len(self.FLAGS.parse_steps) > 0)):
            step=self.FLAGS.parse_steps[0]
            for i in range(self.FLAGS.number_walkers):
                self.assign_weights_and_biases_from_file(self.FLAGS.parse_parameters_file, step,
                                                         walker_index=i, do_check=True)

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
                self.trajectory_writer = setup_trajectory_file(self.FLAGS.trajectory_file,
                                                               self.weights[0].get_total_dof(),
                                                               self.biases[0].get_total_dof(),
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
                header += ['ensemble_average_loss', 'average_kinetic_energy', 'average_virials']
            elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
                header += ['ensemble_average_loss', 'average_kinetic_energy', 'average_virials', 'average_rejection_rate']
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
        elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
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
                for zero assigners, dict with assigners (required for HMC)
        """
        static_vars_float = ["old_loss", "old_kinetic", "kinetic_energy", "total_energy", "momenta", \
                             "gradients", "virials", "noise"]
        static_vars_int64 = ["accepted", "rejected"]
        static_var_dict = {}
        zero_assigner_dict = {}
        assigner_dict = {}
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

                    assigner_loss = static_var_dict["old_loss"][i].assign(self.nn[i].get("loss"))
                    model._dict_append(assigner_dict, "old_loss", assigner_loss)
                    assigner_kinetic = static_var_dict["old_kinetic"][i].assign(static_var_dict["kinetic_energy"][i])
                    model._dict_append(assigner_dict, "old_kinetic", assigner_kinetic)
                    assigner_total = static_var_dict["total_energy"][i].assign(
                        tf.add(self.nn[i].get("loss"), static_var_dict["kinetic_energy"][i]))
                    model._dict_append(assigner_dict, "total_energy", assigner_total)

        return static_var_dict, zero_assigner_dict, assigner_dict

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
        self.init_files("sample")

        placeholder_nodes = [self.nn[walker_index].get_dict_of_nodes(
            ["current_step", "next_eval_step"])
            for walker_index in range(self.FLAGS.number_walkers)]

        list_of_nodes = ["sample_step", "accuracy", "global_step", "loss"]
        if self.FLAGS.summaries_path is not None:
            test_nodes = [self.summary]*self.FLAGS.number_walkers
        else:
            test_nodes = []
        for item in list_of_nodes:
            test_nodes.append([self.nn[walker_index].get(item) \
                               for walker_index in range(self.FLAGS.number_walkers)])

        all_weights = []
        all_biases = []
        for walker_index in range(self.FLAGS.number_walkers):
            all_weights.append(self.weights[walker_index].parameters)
            all_biases.append(self.biases[walker_index].parameters)

        output_width = 8
        output_precision = 8

        written_row = 0

        accumulated_kinetic_energy = [0.]*self.FLAGS.number_walkers
        accumulated_loss_nominator = [0.]*self.FLAGS.number_walkers
        accumulated_loss_denominator = [0.]*self.FLAGS.number_walkers
        accumulated_virials = [0.]*self.FLAGS.number_walkers

        averages = None
        if return_averages:
            averages = []
            steps = (self.FLAGS.max_steps % self.FLAGS.every_nth) + 1
            header = self.get_averages_header(setup="sample")
            no_params = len(header)
            for walker_index in range(self.FLAGS.number_walkers):
                averages.append(pd.DataFrame(
                    np.zeros((steps, no_params)),
                    columns=header))

        run_info = None
        if return_run_info:
            run_info = []
            steps = (self.FLAGS.max_steps % self.FLAGS.every_nth) + 1
            header = self.get_sample_header()
            no_params = len(header)
            for walker_index in range(self.FLAGS.number_walkers):
                run_info.append(pd.DataFrame(
                    np.zeros((steps, no_params)),
                    columns=header))

        trajectory = None
        if return_trajectories:
            trajectory = []
            steps = (self.FLAGS.max_steps % self.FLAGS.every_nth) + 1
            header = get_trajectory_header(
                self.weights[0].get_total_dof(),
                self.biases[0].get_total_dof())
            no_params = len(header)
            for walker_index in range(self.FLAGS.number_walkers):
                trajectory.append(pd.DataFrame(
                    np.zeros((steps, no_params)),
                    columns=header))

        # place in feed dict: We have to supply all placeholders (regardless of
        # which the employed sampler actually requires) because of the evaluated
        # summary! All of the placeholder nodes are also summary nodes.
        feed_dict = {}
        for walker_index in range(self.FLAGS.number_walkers):
            feed_dict.update(self._create_default_feed_dict_with_constants(walker_index))

        # check that sampler's parameters are actually used
        for walker_index in range(self.FLAGS.number_walkers):
            logging.info("Dependent walker #"+str(walker_index))
            if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                      "GeometricLangevinAlgorithm_1stOrder",
                                      "GeometricLangevinAlgorithm_2ndOrder",
                                      "BAOAB",
                                      "CovarianceControlledAdaptiveLangevinThermostat"]:
                gamma, beta, deltat = self.sess.run(self.nn[walker_index].get_list_of_nodes(
                    ["friction_constant", "inverse_temperature", "step_width"]), feed_dict=feed_dict)
                logging.info("LD Sampler parameters, walker #%d: gamma = %lg, beta = %lg, delta t = %lg" %
                      (walker_index, gamma, beta, deltat))
            elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
                current_step, num_mc_steps, deltat = self.sess.run(self.nn[walker_index].get_list_of_nodes(
                    ["current_step", "next_eval_step", "step_width"]), feed_dict=feed_dict)
                logging.info("MC Sampler parameters, walker #%d: current_step = %lg, num_mc_steps = %lg, delta t = %lg" %
                      (walker_index, current_step, num_mc_steps, deltat))

        # create extra nodes for HMC
        if self.FLAGS.sampler == "HamiltonianMonteCarlo":
            HMC_eval_nodes = [[self.nn[i].get("loss") for i in range(self.FLAGS.number_walkers)]] \
                            +[self.static_vars[key] for key in ["old_kinetic", "total_energy"]]

            HMC_set_nodes = []
            HMC_set_nodes.extend([self.assigner["old_loss"],
                                  self.assigner["old_kinetic"]])
            HMC_set_all_nodes = []
            HMC_set_all_nodes.extend([self.assigner["total_energy"]] + HMC_set_nodes)

            # zero rejection rate before sampling start
            check_accepted, check_rejected = self.sess.run([
                self.zero_assigner["accepted"], self.zero_assigner["rejected"]])
            for walker_index in range(self.FLAGS.number_walkers):
                assert(check_accepted[walker_index] == 0)
                assert(check_rejected[walker_index] == 0)

        # prepare summaries for TensorBoard
        if self.FLAGS.summaries_path is not None:
            summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_path, self.sess.graph)

        logging.info("Starting to sample")
        logging.info_intervals = max(1, int(self.FLAGS.max_steps / 100))
        last_time = time.process_time()
        elapsed_time = 0.
        HMC_steps = [0]*self.FLAGS.number_walkers
        HMC_current_set_nodes = []
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

            # set HMC specific nodes in feed_dict
            if self.FLAGS.sampler == "HamiltonianMonteCarlo":
                for walker_index in range(self.FLAGS.number_walkers):
                    # pick next evaluation step with a little random variation
                    if current_step > HMC_steps[walker_index]:
                        HMC_steps[walker_index] += max(1,
                                         round((0.9 + np.random.uniform(low=0., high=0.2)) \
                                               * self.FLAGS.hamiltonian_dynamics_time / self.FLAGS.step_width))
                        logging.debug("Next evaluation of HMC criterion at step " + str(HMC_steps))
                        feed_dict.update({
                            placeholder_nodes[walker_index]["next_eval_step"]: HMC_steps[walker_index]
                        })
                    feed_dict.update({
                        placeholder_nodes[walker_index]["current_step"]: current_step
                    })

            # set global variable used in HMC sampler for criterion to initial loss
            if self.FLAGS.sampler == "HamiltonianMonteCarlo":
                loss_eval, kin_eval, total_eval = self.sess.run(HMC_eval_nodes, feed_dict=feed_dict)
                for walker_index in range(self.FLAGS.number_walkers):
                    logging.debug("walker #%d, #%d: loss is %lg, total is %lg, kinetic is %lg" \
                                  % (walker_index, current_step, loss_eval[walker_index], total_eval[walker_index], kin_eval[walker_index]))
                HMC_current_set_nodes[:] = HMC_set_nodes
                for walker_index in range(self.FLAGS.number_walkers):
                    if abs(total_eval[walker_index]) < 1e-10:
                        HMC_current_set_nodes.extend([self.assigner["total_energy"][walker_index]])
                self.sess.run(HMC_current_set_nodes, feed_dict=feed_dict)
                loss_eval, kin_eval, total_eval = self.sess.run(HMC_eval_nodes, feed_dict=feed_dict)
                for walker_index in range(self.FLAGS.number_walkers):
                    logging.debug("walker #%d, #%d: loss is %lg, total is %lg, kinetic is %lg" \
                                  % (walker_index, current_step, loss_eval[walker_index], total_eval[walker_index], kin_eval[walker_index]))

            # zero kinetic energy
            if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                      "GeometricLangevinAlgorithm_1stOrder",
                                      "GeometricLangevinAlgorithm_2ndOrder",
                                      "HamiltonianMonteCarlo",
                                      "BAOAB",
                                      "CovarianceControlledAdaptiveLangevinThermostat"]:
                check_kinetic, check_momenta, check_gradients, check_virials, check_noise = \
                    self.sess.run([
                        self.zero_assigner["kinetic_energy"],
                        self.zero_assigner["momenta"],
                        self.zero_assigner["gradients"],
                        self.zero_assigner["virials"],
                        self.zero_assigner["noise"]])
                for walker_index in range(self.FLAGS.number_walkers):
                    assert (abs(check_kinetic[walker_index]) < 1e-10)
                    assert (abs(check_momenta[walker_index]) < 1e-10)
                    assert (abs(check_gradients[walker_index]) < 1e-10)
                    assert (abs(check_virials[walker_index]) < 1e-10)
                    assert (abs(check_noise[walker_index]) < 1e-10)

            # get the weights and biases as otherwise the loss won't match
            # tf first computes loss, then gradient, then performs variable update
            # henc+ after the sample step, we would have updated variables but old loss
            if current_step % self.FLAGS.every_nth == 0:
                if self.config_map["do_write_trajectory_file"] or return_trajectories:
                    weights_eval, biases_eval = self.sess.run([all_weights, all_biases])
                    #[logging.info(str(item)) for item in weights_eval]
                    #[logging.info(str(item)) for item in biases_eval]

            # NOTE: All values from nodes contained in the same call to tf.run() with train_step
            # will be evaluated as if before train_step. Nodes that are changed in the update due to
            # train_step (e.g. momentum_t) however are updated.
            # In other words, whether we use
            #   tf.run([train_step, loss_eval], ...)
            # or
            #   tf.run([loss_eval, train_step], ...)
            # is not important. Only a subsequent, distinct tf.run() call would produce a different loss_eval.
            if self.FLAGS.summaries_path is not None:
                summary, _, acc, global_step, loss_eval = \
                    self.sess.run(test_nodes, feed_dict=feed_dict)
            else:
                _, acc, global_step, loss_eval = \
                    self.sess.run(test_nodes, feed_dict=feed_dict)

            if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                      "GeometricLangevinAlgorithm_1stOrder",
                                      "GeometricLangevinAlgorithm_2ndOrder",
                                      "HamiltonianMonteCarlo",
                                      "BAOAB",
                                      "CovarianceControlledAdaptiveLangevinThermostat"]:
                if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                    gradients, virials, noise = \
                        self.sess.run([
                            self.static_vars["gradients"],
                            self.static_vars["virials"],
                            self.static_vars["noise"]])
                elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
                    old_total_energy, kinetic_energy, momenta, gradients, virials = \
                        self.sess.run([
                            self.static_vars["total_energy"],
                            self.static_vars["kinetic_energy"],
                            self.static_vars["momenta"],
                            self.static_vars["gradients"],
                            self.static_vars["virials"]])
                else:
                    kinetic_energy, momenta, gradients, virials, noise = \
                        self.sess.run([
                            self.static_vars["kinetic_energy"],
                            self.static_vars["momenta"],
                            self.static_vars["gradients"],
                            self.static_vars["virials"],
                            self.static_vars["noise"]])

            if self.FLAGS.summaries_path is not None:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary_writer.add_run_metadata(run_metadata, 'step%d' % current_step)
                summary_writer.add_summary(summary, current_step)

            for walker_index in range(self.FLAGS.number_walkers):
                if current_step >= self.FLAGS.burn_in_steps:
                    accumulated_loss_nominator[walker_index] += loss_eval[walker_index] * exp(- self.FLAGS.inverse_temperature * loss_eval[walker_index])
                    accumulated_loss_denominator[walker_index] += exp(- self.FLAGS.inverse_temperature * loss_eval[walker_index])
                    if self.FLAGS.sampler != "StochasticGradientLangevinDynamics":
                        accumulated_kinetic_energy[walker_index] += kinetic_energy[walker_index]
                    accumulated_virials[walker_index] += virials[walker_index]

            if current_step % self.FLAGS.every_nth == 0:
                current_time = time.process_time()
                time_elapsed_per_nth_step = current_time - last_time
                if current_step > 1:
                    elapsed_time += time_elapsed_per_nth_step
                    estimated_time_left = (self.FLAGS.max_steps-current_step)*elapsed_time/(current_step-1)
                else:
                    estimated_time_left = 0.
                logging.debug("Output at step #" + str(current_step) \
                              + ", est. remaining time is " + str(estimated_time_left)+" seconds.")
                last_time = current_time

                if self.FLAGS.sampler == "HamiltonianMonteCarlo":
                    accepted_eval, rejected_eval = self.sess.run([
                        self.static_vars["accepted"], self.static_vars["rejected"]])

                for walker_index in range(self.FLAGS.number_walkers):
                    if self.config_map["do_write_trajectory_file"] or return_trajectories:
                        trajectory_line = [walker_index, global_step[walker_index]] \
                                          + ['{:{width}.{precision}e}'.format(loss_eval[walker_index], width=output_width,
                                                                              precision=output_precision)]
                        if len(weights_eval[walker_index]) > 0:
                            flat_array = neuralnet_parameters.flatten_list_of_arrays(weights_eval[walker_index])
                            trajectory_line += ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision) \
                                                for item in flat_array[:]]
                        if len(biases_eval[walker_index]) > 0:
                            flat_array = neuralnet_parameters.flatten_list_of_arrays(biases_eval[walker_index])
                            trajectory_line += ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision) \
                                                for item in flat_array[:]]

                        if self.config_map["do_write_trajectory_file"]:
                            self.trajectory_writer.writerow(trajectory_line)
                        if return_trajectories:
                            trajectory[walker_index].loc[written_row] = trajectory_line

                    if self.config_map["do_write_averages_file"] or return_averages:
                        if accumulated_loss_denominator[walker_index] > 0:
                            average_loss = accumulated_loss_nominator[walker_index]/accumulated_loss_denominator[walker_index]
                        else:
                            average_loss = 0.
                        if current_step >= self.FLAGS.burn_in_steps:
                            divisor = float(current_step + 1. - self.FLAGS.burn_in_steps)
                            average_kinetic_energy = accumulated_kinetic_energy[walker_index] / divisor
                            average_virials = abs(0.5 * accumulated_virials[walker_index]) / divisor
                        else:
                            average_kinetic_energy = 0.
                            average_virials = 0.
                        averages_line = [walker_index, global_step[walker_index], current_step] \
                                        + ['{:{width}.{precision}e}'.format(loss_eval[walker_index], width=output_width,
                                                                            precision=output_precision)] \
                                        + ['{:{width}.{precision}e}'.format(average_loss, width=output_width,
                                                                            precision=output_precision)]
                        if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                            averages_line += ['{:{width}.{precision}e}'.format(average_virials, width=output_width,
                                                                               precision=output_precision)]
                        else:
                            averages_line += ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                           precision=output_precision)
                                          for x in [average_kinetic_energy,average_virials]]

                        if self.config_map["do_write_averages_file"]:
                            self.averages_writer.writerow(averages_line)
                        if return_averages:
                            averages[walker_index].loc[written_row] = averages_line

                    if self.config_map["do_write_run_file"] or return_run_info:
                        run_line  = []
                        if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                                  "GeometricLangevinAlgorithm_1stOrder",
                                                  "GeometricLangevinAlgorithm_2ndOrder",
                                                  "HamiltonianMonteCarlo",
                                                  "BAOAB",
                                                  "CovarianceControlledAdaptiveLangevinThermostat"]:
                            run_line = [walker_index, global_step[walker_index], current_step] \
                                       + ['{:1.3f}'.format(acc[walker_index])] \
                                       + ['{:{width}.{precision}e}'.format(loss_eval[walker_index], width=output_width,
                                                                           precision=output_precision)] \
                                       + ['{:{width}.{precision}e}'.format(time_elapsed_per_nth_step, width=output_width,
                                                                           precision=output_precision)]
                            if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                                run_line += ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                              precision=output_precision)
                                             for x in [sqrt(gradients[walker_index]), abs(0.5*virials[walker_index]), sqrt(noise[walker_index])]]
                            elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
                                if (rejected_eval[walker_index]+accepted_eval[walker_index]) > 0:
                                    rejection_rate = rejected_eval[walker_index]/(rejected_eval[walker_index]+accepted_eval[walker_index])
                                else:
                                    rejection_rate = 0
                                run_line += ['{:{width}.{precision}e}'.format(loss_eval[walker_index] + kinetic_energy[walker_index],
                                                                              width=output_width,
                                                                              precision=output_precision)]\
                                           + ['{:{width}.{precision}e}'.format(old_total_energy[walker_index],
                                                                               width=output_width,
                                                                               precision=output_precision)]\
                                           + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                               precision=output_precision)
                                              for x in [kinetic_energy[walker_index], sqrt(momenta[walker_index]), sqrt(gradients[walker_index]), abs(0.5*virials[walker_index])]]\
                                           + ['{:{width}.{precision}e}'.format(rejection_rate, width=output_width,
                                                                               precision=output_precision)]
                            else:
                                run_line += ['{:{width}.{precision}e}'.format(loss_eval[walker_index] + kinetic_energy[walker_index],
                                                                              width=output_width,
                                                                              precision=output_precision)]\
                                           + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                               precision=output_precision)
                                              for x in [kinetic_energy[walker_index], sqrt(momenta[walker_index]), sqrt(gradients[walker_index]), abs(0.5*virials[walker_index]), sqrt(noise[walker_index])]]

                        if self.config_map["do_write_run_file"]:
                            self.run_writer.writerow(run_line)
                        if return_run_info:
                            run_info[walker_index].loc[written_row] = run_line

            if current_step % self.FLAGS.every_nth == 0:
                written_row+=1

            #if (i % logging.info_intervals) == 0:
                #logging.debug('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                #logging.debug('Loss at step %s: %s' % (i, loss_eval))

            if (self.FLAGS.number_walkers > 1) and (self.FLAGS.collapse_after_steps != 0) and \
                    (current_step % self.FLAGS.collapse_after_steps == 0):
                # get walker 0's position
                weights_eval, biases_eval = self.sess.run([
                    self.weights[0].parameters, self.biases[0].parameters])

                # reset positions of walker 1 till end to that of walker 0
                # assign all in a single session run to allow parallelization
                collapse_feed_dict = {}
                assigns = []
                for walker_index in range(1,self.FLAGS.number_walkers):
                    # directly connecting the flat parameters tensor with the respective
                    # other walker's parameters' placeholder does not seem to work, i.e.
                    # replacing weights_eval -> self.weights[0].parameters
                    assert( len(self.weights[0].parameters) == len(self.weights[walker_index].placeholders))
                    for weight, weight_placeholder in zip(weights_eval,
                                                          self.weights[walker_index].placeholders):
                        collapse_feed_dict[weight_placeholder] = weight
                    assert( len(self.biases[0].parameters) == len(self.biases[walker_index].placeholders))
                    for bias, bias_placeholder in zip(biases_eval,
                                                      self.biases[walker_index].placeholders):
                        collapse_feed_dict[bias_placeholder] = bias
                    assigns.append(self.weights[walker_index].assign_all_t)
                    assigns.append(self.biases[walker_index].assign_all_t)
                # evaluate and assign all at once
                self.sess.run(assigns, feed_dict=collapse_feed_dict)

        logging.info("SAMPLED.")

        # close summaries file
        if self.FLAGS.summaries_path is not None:
            summary_writer.close()

        self.finish_files()

        return run_info, trajectory, averages

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
        for key in ["covariance_blending", "friction_constant", "inverse_temperature",
                    "learning_rate", "sigma", "sigmaA", "step_width"]:
            try:
                param_dict[key] = getattr(self.FLAGS, key)
            except AttributeError:
                pass
        # special case because key and attribute's name differ
        try:
            param_dict["next_eval_step"] = self.FLAGS.hamiltonian_dynamics_time
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

        placeholder_nodes = self.nn[walker_index].get_dict_of_nodes(["learning_rate", "y_"])
        if self.FLAGS.summaries_path is not None:
            test_nodes = [self.summary]
        else:
            test_nodes = []
        test_nodes.extend(self.nn[walker_index].get_list_of_nodes(
            ["train_step", "accuracy", "global_step", "loss", "y_", "y"]) \
                     +[self.static_vars["gradients"]])

        output_width = 8
        output_precision = 8

        written_row = 0

        accumulated_virials = 0.

        averages = None
        if return_averages:
            steps = (self.FLAGS.max_steps % self.FLAGS.every_nth)+1
            header = self.get_averages_header(setup="train")
            no_params = len(header)
            averages = pd.DataFrame(
                np.zeros((steps, no_params)),
                columns=header)

        run_info = None
        if return_run_info:
            steps = (self.FLAGS.max_steps % self.FLAGS.every_nth)+1
            header = self.get_train_header()
            no_params = len(header)
            run_info = pd.DataFrame(
                np.zeros((steps, no_params)),
                columns=header)

        trajectory = None
        if return_trajectories:
            steps = (self.FLAGS.max_steps % self.FLAGS.every_nth)+1
            header = get_trajectory_header(
                self.weights[0].get_total_dof(),
                self.biases[0].get_total_dof())
            no_params = self.weights[0].get_total_dof()+self.biases[0].get_total_dof()+3
            trajectory = pd.DataFrame(
                np.zeros((steps, no_params)),
                columns=header)


        default_feed_dict = self._create_default_feed_dict_with_constants(walker_index)

        # prepare summaries for TensorBoard
        if self.FLAGS.summaries_path is not None:
            summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_path, self.sess.graph)

        logging.info("Starting to train")
        last_time = time.process_time()
        elapsed_time = 0
        if tqdm_present and self.FLAGS.progress:
            step_range = tqdm(range(self.FLAGS.max_steps))
        else:
            step_range = range(self.FLAGS.max_steps)
        for current_step in step_range:
            logging.debug("Current step is " + str(current_step))

            # get next batch of data
            features, labels = self.input_pipeline.next_batch(self.sess)

            # place in feed dict
            feed_dict = default_feed_dict.copy()
            feed_dict.update({
                self.xinput: features,
                self.true_labels: labels
            })
            #logging.debug("batch is x: "+str(features[:])+", y: "+str(labels[:]))

            # zero accumulated gradient
            check_gradients, check_virials = self.sess.run([self.zero_assigner["gradients"][walker_index],
                                                            self.zero_assigner["virials"][walker_index]])
            assert (abs(check_gradients) < 1e-10)
            assert (abs(check_virials) < 1e-10)

            # get the weights and biases as otherwise the loss won't match
            # tf first computes loss, then gradient, then performs variable update
            # hence, after the sample step, we would have updated variables but old loss
            if current_step % self.FLAGS.every_nth == 0:
                if self.config_map["do_write_trajectory_file"] or return_trajectories:
                    weights_eval = self.weights[walker_index].evaluate(self.sess)
                    biases_eval = self.biases[walker_index].evaluate(self.sess)

            if self.FLAGS.summaries_path is not None:
                summary, _, acc, global_step, loss_eval, y_true_eval, y_eval, scaled_grad = \
                    self.sess.run(test_nodes, feed_dict=feed_dict)
            else:
                _, acc, global_step, loss_eval, y_true_eval, y_eval, scaled_grad = \
                    self.sess.run(test_nodes, feed_dict=feed_dict)

            gradients, virials = self.sess.run([self.static_vars["gradients"][walker_index],
                                                self.static_vars["virials"][walker_index]])
            if current_step >= self.FLAGS.burn_in_steps:
                accumulated_virials += virials

            if self.FLAGS.summaries_path is not None:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary_writer.add_run_metadata(run_metadata, 'step%d' % current_step)
                summary_writer.add_summary(summary, current_step)

            # we always write last step
            if (current_step % self.FLAGS.every_nth == 0) or (current_step == (self.FLAGS.max_steps-1)):
                current_time = time.process_time()
                time_elapsed_per_nth_step = current_time - last_time
                # neglect first step as initialization perturbs average
                if current_step > 1:
                    elapsed_time += time_elapsed_per_nth_step
                    estimated_time_left = (self.FLAGS.max_steps-current_step)*elapsed_time/(current_step-1)
                else:
                    estimated_time_left = 0.
                logging.debug("Output at step #" + str(current_step) \
                              + ", est. remaining time is " + str(estimated_time_left)+" seconds.")
                last_time = current_time

                run_line = [0, global_step, current_step] + ['{:1.3f}'.format(acc)] \
                           + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                               precision=output_precision)] \
                           + ['{:{width}.{precision}e}'.format(time_elapsed_per_nth_step, width=output_width,
                                                               precision=output_precision)] \
                           + ['{:{width}.{precision}e}'.format(sqrt(gradients), width=output_width,
                                                               precision=output_precision)] \
                           + ['{:{width}.{precision}e}'.format(abs(0.5*virials),width=output_width,
                                                               precision=output_precision)]
                if self.config_map["do_write_run_file"]:
                    self.run_writer.writerow(run_line)
                if return_run_info:

                    run_info.loc[written_row] = run_line
                if self.config_map["do_write_averages_file"] or return_averages:
                    if current_step >= self.FLAGS.burn_in_steps:
                        average_virials = abs(0.5 * accumulated_virials) / (float(current_step + 1. - self.FLAGS.burn_in_steps))
                    else:
                        average_virials = 0.
                    averages_line = [0, global_step, current_step] \
                                    + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                                        precision=output_precision)] \
                                    + ['{:{width}.{precision}e}'.format(average_virials, width=output_width,
                                                                        precision=output_precision)]

                    if self.config_map["do_write_averages_file"]:
                        self.averages_writer.writerow(averages_line)
                    if return_averages:
                        averages.loc[written_row] = averages_line


                if return_trajectories or self.config_map["do_write_trajectory_file"]:
                    trajectory_line = [0, global_step] \
                                      + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                                          precision=output_precision)] \
                                      + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                                         for item in weights_eval] \
                                      + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                                         for item in biases_eval]
                    if self.config_map["do_write_trajectory_file"]:
                        self.trajectory_writer.writerow(trajectory_line)
                    if return_trajectories:
                        trajectory.loc[written_row] = trajectory_line
                written_row+=1

            logging.debug('Accuracy at step %s (%s): %s' % (current_step, global_step, acc))
            logging.debug('Loss at step %s: %s' % (current_step, loss_eval))
            #logging.debug('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
            #logging.debug('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
        logging.info("TRAINED down to loss %s and accuracy %s." % (loss_eval, acc))

        # close summaries file
        if self.FLAGS.summaries_path is not None:
            summary_writer.close()

        self.finish_files()

        return run_info, trajectory, averages

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
        self.weights[walker_index].assign(self.sess, weights_vals)
        self.biases[walker_index].assign(self.sess, biases_vals)

        # get the input and biases to check against what we set
        if do_check:
            weights_eval = self.weights[walker_index].evaluate(self.sess)
            biases_eval = self.biases[walker_index].evaluate(self.sess)
            logging.info("Evaluating walker #"+str(walker_index) \
                         +" at weights " + str(weights_eval[0:10]) \
                         + ", biases " + str(biases_eval[0:10]))
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
        # parse csv file
        parameters = {}
        for keyname in df_parameters.columns:
            if (keyname[1] >= "0" and keyname[1] <= "9"):
                if ("w" == keyname[0]):
                    fullname = "weight"
                elif "b" == keyname[0]:
                    fullname = "bias"
                else:
                    # not a parameter column
                    continue
                fullname += keyname[1:]
                parameters[fullname] = df_parameters.loc[rownr, [keyname]].values[0]
            else:
                if ("weight" in keyname) or ("bias" in keyname):
                    parameters[keyname] = df_parameters.loc[rownr, [keyname]].values[0]
        logging.debug("Read row "+str(rownr)+":"+str([parameters[key] for key in ["weight0", "weight1", "weight2"] if key in parameters.keys()]) \
                      +"..."+str([parameters[key] for key in ["bias0", "bias1", "bias2"] if key in parameters.keys()]))

        # create internal array to store parameters
        weights_vals = self.weights[walker_index].create_flat_vector()
        biases_vals = self.biases[walker_index].create_flat_vector()
        weights_vals[:weights_vals.size] = [parameters[key] for key in sorted(parameters.keys()) if "w" in key]
        biases_vals[:biases_vals.size] = [parameters[key] for key in sorted(parameters.keys()) if "b" in key]
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
        if step in df_parameters.loc[:, ['step']].values:
            rownr = np.where(df_parameters.loc[:, ['step']].values == step)[0]
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
