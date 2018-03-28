from builtins import staticmethod

import logging
from math import sqrt, exp
import numpy as np
import pandas as pd
import scipy.sparse as sps
import tensorflow as tf
import time

from TATi.common import create_input_layer, file_length, get_list_from_string, \
    get_trajectory_header, initialize_config_map, setup_csv_file, setup_run_file, \
    setup_trajectory_file
from TATi.models.input.datasetpipeline import DatasetPipeline
from TATi.models.input.inmemorypipeline import InMemoryPipeline
from TATi.models.basetype import dds_basetype
from TATi.models.mock_flags import MockFlags
from TATi.models.neuralnet_parameters import neuralnet_parameters
from TATi.models.neuralnetwork import NeuralNetwork


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
        self.config_map = initialize_config_map()

        try:
            FLAGS.max_steps
        except AttributeError:
            FLAGS.max_steps = 1

        self.number_of_parameters = 0 # number of biases and weights

        if len(FLAGS.batch_data_files) > 0:
            self.input_dimension = self.FLAGS.input_dimension
            self.output_dimension = self.FLAGS.output_dimension
            if FLAGS.batch_data_file_type == "csv":
                self.FLAGS.dimension = sum([file_length(filename)
                                            for filename in FLAGS.batch_data_files]) \
                                       - len(FLAGS.batch_data_files)
                self._check_valid_batch_size()
            elif FLAGS.batch_data_file_type == "tfrecord":
                self.FLAGS.dimension = self._get_dimension_from_tfrecord(FLAGS.batch_data_files)
            else:
                logging.info("Unknown file type")
                assert(0)

            logging.info("Parsing "+str(FLAGS.batch_data_files))

            self.number_of_parameters = 0 # number of biases and weights

        # mark input layer as to be created
        self.xinput = None
        self.x = None

        # mark resource variables as to be created
        self.resources_created = None

        # mark already fixes variables
        self.fixed_variables = []

        # mark neuralnetwork, saver and session objects as to be created
        self.nn = None
        self.saver = None
        self.sess = None

        # mark placeholder neuralnet_parameters as to be created
        self.weights = None
        self.biases = None

        # mark placeholders for gradient and hessian computation as to be created
        self.gradients = None
        self.hessians = None

        # mark step assign op as to be created
        self.global_step_assign_t = None

        # mark writer as to be created
        self.averages_writer = None
        self.run_writer = None
        self.trajectory_writer = None

    def init_input_pipeline(self):
        self.batch_next = self.create_input_pipeline(self.FLAGS)
        self.input_pipeline.reset(self.sess)

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
                    logging.debug("height is "+str(height)+" and width is "+str(width))
                dimension += 1

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
        assert( len(features) == len(labels) )
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
        if FLAGS.in_memory_pipeline and (FLAGS.batch_data_file_type == "csv"):
            logging.info("Using in-memory pipeline")
            # at the moment we can only parse a single file
            assert( len(FLAGS.batch_data_files) == 1 )
            csv_dataset = pd.read_csv(FLAGS.batch_data_files[0], sep=',', header=0)
            xs = np.asarray(csv_dataset.iloc[:, 0:self.input_dimension])
            ys = np.asarray(csv_dataset.iloc[:, self.input_dimension:self.input_dimension+self.output_dimension])
            self.input_pipeline = InMemoryPipeline(dataset=[xs,ys], batch_size=FLAGS.batch_size,
                                                   max_steps=FLAGS.max_steps,
                                                   shuffle=shuffle, seed=FLAGS.seed)
        else:
            logging.info("Using tf.Dataset pipeline")
            self.input_pipeline = DatasetPipeline(filenames=FLAGS.batch_data_files, filetype=FLAGS.batch_data_file_type,
                                                  batch_size=FLAGS.batch_size, dimension=FLAGS.dimension, max_steps=FLAGS.max_steps,
                                                  input_dimension=self.input_dimension, output_dimension=self.output_dimension,
                                                  shuffle=shuffle, seed=FLAGS.seed)

    def reset_parameters(self, FLAGS):
        """ Use to pass a different set of FLAGS controlling training or sampling.

        :param FLAGS: new set of parameters
        """
        self.FLAGS = FLAGS

    def create_resource_variables(self):
        """ Creates some global resource variables to hold statistical quantities
        during sampling.
        """
        with tf.variable_scope("accumulate", reuse=self.resources_created):
            # the following are used for HMC
            old_loss_t = tf.get_variable("old_loss", shape=[], trainable=False,
                                               initializer=tf.zeros_initializer,
                                               use_resource=True, dtype=dds_basetype)
            old_kinetic_t = tf.get_variable("old_kinetic", shape=[], trainable=False,
                                               initializer=tf.zeros_initializer,
                                               use_resource=True, dtype=dds_basetype)
            total_energy_t = tf.get_variable("total_energy", shape=[], trainable=False,
                                               initializer=tf.zeros_initializer,
                                               use_resource=True, dtype=dds_basetype)
            # used for Langevin samplers
            kinetic_energy_t = tf.get_variable("kinetic", shape=[], trainable=False,
                                               initializer=tf.zeros_initializer,
                                               use_resource=True, dtype=dds_basetype)
            momenta_t = tf.get_variable("momenta", shape=[], trainable=False,
                                        initializer=tf.zeros_initializer,
                                        use_resource=True, dtype=dds_basetype)
            gradients_t = tf.get_variable("gradients", shape=[], trainable=False,
                                          initializer=tf.zeros_initializer,
                                          use_resource=True, dtype=dds_basetype)
            virials_t = tf.get_variable("virials", shape=[], trainable=False,
                                        initializer=tf.zeros_initializer,
                                        use_resource=True, dtype=dds_basetype)
            noise_t = tf.get_variable("noise", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer,
                                      use_resource=True, dtype=dds_basetype)
            # the following are used for HMC to measure rejection rate
            rejected_t = tf.get_variable("rejected", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer,
                                      use_resource=True, dtype=tf.int64)
            accepted_t = tf.get_variable("accepted", shape=[], trainable=False,
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
    def setup_parameters(
            alpha=1.,
            averages_file=None,
            batch_data_files=[],
            batch_data_file_type="csv",
            batch_size=10,
            diffusion_map_method="vanilla",
            do_hessians=False,
            dropout=None,
            every_nth=1,
            fix_parameters=None,
            friction_constant=0.,
            hidden_activation="relu",
            hidden_dimension="",
            in_memory_pipeline=False,
            input_columns="1 2",
            input_dimension=2,
            inter_ops_threads=1,
            intra_ops_threads=None,
            inverse_temperature=1.,
            inverse_temperature_max=1.,
            loss="mean_squared",
            max_steps=1000,
            number_of_eigenvalues=4,
            number_of_temperatures=10,
            hamiltonian_dynamics_time=10,
            optimizer="GradientDescent",
            output_activation="tanh",
            output_dimension=1,
            parse_parameters_file=None,
            parse_steps=[],
            prior_factor=1.,
            prior_lower_boundary=None,
            prior_power=1.,
            prior_upper_boundary=None,
            restore_model=None,
            run_file=None,
            sampler="GeometricLangevinAlgorithm_1stOrder",
            save_model=None,
            seed=None,
            step_width=0.03,
            summaries_path=None,
            trajectory_file=None,
            use_reweighting=False,
            verbose=0):
            return MockFlags(
                alpha=alpha,
                averages_file=averages_file,
                batch_data_files=batch_data_files,
                batch_data_file_type=batch_data_file_type,
                batch_size=batch_size,
                diffusion_map_method=diffusion_map_method,
                do_hessians=do_hessians,
                dropout=dropout,
                every_nth=every_nth,
                fix_parameters=fix_parameters,
                friction_constant=friction_constant,
                hidden_activation=hidden_activation,
                hidden_dimension=hidden_dimension,
                in_memory_pipeline=in_memory_pipeline,
                input_columns=input_columns,
                input_dimension=input_dimension,
                inter_ops_threads=inter_ops_threads,
                intra_ops_threads=intra_ops_threads,
                inverse_temperature=inverse_temperature,
                inverse_temperature_max=inverse_temperature_max,
                loss=loss,
                max_steps=max_steps,
                number_of_eigenvalues=number_of_eigenvalues,
                number_of_temperatures=number_of_temperatures,
                hamiltonian_dynamics_time=hamiltonian_dynamics_time,
                optimizer=optimizer,
                output_activation=output_activation,
                output_dimension=output_dimension,
                parse_parameters_file=parse_parameters_file,
                parse_steps=parse_steps,
                prior_factor=prior_factor,
                prior_lower_boundary=prior_lower_boundary,
                prior_power=prior_power,
                prior_upper_boundary=prior_upper_boundary,
                restore_model=restore_model,
                run_file=run_file,
                sampler=sampler,
                save_model=save_model,
                seed=seed,
                step_width=step_width,
                summaries_path=summaries_path,
                trajectory_file=trajectory_file,
                use_reweighting=use_reweighting,
                verbose=verbose)

    def reset_dataset(self):
        """ Re-initializes the dataset for a new run
        """
        self.input_pipeline.reset(self.sess)

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

        #if setup == "sample":
        self.create_resource_variables()

        if self.nn is None:
            self.nn = NeuralNetwork()
            hidden_dimension = [int(i) for i in get_list_from_string(self.FLAGS.hidden_dimension)]
            activations = NeuralNetwork.get_activations()
            self.loss = self.nn.create(
                self.x, hidden_dimension, self.output_dimension,
                seed=self.FLAGS.seed,
                add_dropped_layer=(self.FLAGS.dropout is not None),
                hidden_activation=activations[self.FLAGS.hidden_activation],
                output_activation=activations[self.FLAGS.output_activation],
                loss_name=self.FLAGS.loss
            )

            if self.FLAGS.do_hessians or add_vectorized_gradients:
                # create node for gradient and hessian computation only if specifically
                # requested as the creation along is costly (apart from the expensive part
                # of evaluating the nodes eventually). This would otherwise slow down
                # startup quite a bit even when hessians are not evaluated.
                #print("GRADIENTS")
                vectorized_gradients = []
                for tensor in tf.get_collection(tf.GraphKeys.WEIGHTS) + tf.get_collection(tf.GraphKeys.BIASES):
                    grad = tf.gradients(self.loss, tensor)
                    print(grad)
                    vectorized_gradients.append(tf.reshape(grad, [-1]))
                self.gradients = tf.reshape(tf.concat(vectorized_gradients, axis=0), [-1])

            if self.FLAGS.do_hessians:
                #print("HESSIAN")
                self.hessians = []
                total_dofs = 0
                for gradient in vectorized_gradients:
                    dofs = int(np.cumprod(gradient.shape))
                    total_dofs += dofs
                    #print(dofs)
                    # tensorflow cannot compute the gradient of a multi-dimensional mapping
                    # only of functions (i.e. one-dimensional output). Hence, we have to
                    # split the gradients into its components and do gradient on each
                    split_gradient = tf.split(gradient, num_or_size_splits=dofs)
                    for splitgrad in split_gradient:
                        for othertensor in tf.get_collection(tf.GraphKeys.WEIGHTS) + tf.get_collection(tf.GraphKeys.BIASES):
                            grad = tf.gradients(splitgrad, othertensor)
                            self.hessians.append(
                                tf.reshape(grad, [-1]))
                self.hessians = tf.reshape(tf.concat(self.hessians, axis=0), [total_dofs, total_dofs])
        else:
            self.loss = self.nn.get_list_of_nodes(["loss"])[0]

        if self.FLAGS.fix_parameters is not None:
            names, values = self.split_parameters_as_names_values(self.FLAGS.fix_parameters)
            fixed_variables = self.fix_parameters(names)
            logging.debug("Excluded the following degrees of freedom: "+str(fixed_variables))

        # set number of degrees of freedom
        self.number_of_parameters = \
            neuralnet_parameters.get_total_dof_from_list(
                tf.get_collection(tf.GraphKeys.WEIGHTS)) \
            + neuralnet_parameters.get_total_dof_from_list(
                tf.get_collection(tf.GraphKeys.BIASES))

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
        if setup == "train":
            self.nn.add_train_method(self.loss, optimizer_method=self.FLAGS.optimizer, prior=prior)
        elif setup == "sample":
            self.nn.add_sample_method(self.loss, sampling_method=self.FLAGS.sampler, seed=self.FLAGS.seed,
                                      inverse_temperature_max=self.FLAGS.inverse_temperature_max,
                                      number_of_temperatures=self.FLAGS.number_of_temperatures,
                                      alpha_constant=self.FLAGS.alpha,
                                      prior=prior)
        else:
            logging.info("Not adding sample or train method.")

        if "step_placeholder" not in self.nn.placeholder_nodes.keys():
            step_placeholder = tf.placeholder(shape=(), dtype=tf.int32)
            self.nn.placeholder_nodes["step_placeholder"] = step_placeholder
        if (self.global_step_assign_t is None) and ('global_step' in self.nn.summary_nodes.keys()):
            self.global_step_assign_t = tf.assign(self.nn.summary_nodes['global_step'],
                                                  self.nn.placeholder_nodes["step_placeholder"])

        # setup model saving/recovering
        if self.saver is None:
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.WEIGHTS) +
                                   tf.get_collection(tf.GraphKeys.BIASES) + \
                                   tf.get_collection("Variables_to_Save"))
        if self.sess is None:
            logging.debug("Using %s, %s threads " % (str(self.FLAGS.intra_ops_threads), str(self.FLAGS.inter_ops_threads)))
            self.sess = tf.Session(
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=self.FLAGS.intra_ops_threads,
                    inter_op_parallelism_threads=self.FLAGS.inter_ops_threads))

        # initialize constants in graph
        self.nn.init_graph(self.sess)

        # initialize dataset
        #self.input_pipeline.reset(self.sess)

        if self.FLAGS.fix_parameters is not None:
            logging.debug("Assigning the following values to fixed degrees of freedom: "+str(values))
            self.assign_parameters(fixed_variables, values)

        # assign state of model from file if given
        if filename is not None:
            # Tensorflow DOCU says: initializing is not needed when restoring
            # however, global_variables are missing otherwise for storing kinetic, ...
            # tf.reset_default_graph()

            restore_path = filename.replace('.meta', '')
            self.saver.restore(self.sess, restore_path)
            logging.info("Model restored from file: %s" % restore_path)

        if self.weights is None:
            self.weights = neuralnet_parameters(tf.get_collection(tf.GraphKeys.WEIGHTS))
        if self.biases is None:
            self.biases = neuralnet_parameters(tf.get_collection(tf.GraphKeys.BIASES))

        # assign parameters of NN from step in given file
        if self.FLAGS.parse_parameters_file is not None \
                and (self.FLAGS.parse_steps is not None and (len(self.FLAGS.parse_steps) > 0)):
            step=self.FLAGS.parse_steps[0]
            self.assign_weights_and_biases_from_file(self.FLAGS.parse_parameters_file, step, do_check=True)

        header = None
        logging.info("Setting up output files for "+str(setup))
        if setup == "sample":
            header = self.get_sample_header()
        elif setup == "train":
            header = self.get_train_header()

        # merge summaries at very end
        merged = tf.summary.merge_all()  # Merge all the summaries
        self.nn.summary_nodes['merged'] = merged

        try:
            if self.averages_writer is None:
                if self.FLAGS.averages_file is not None:
                    self.config_map["do_write_averages_file"] = True
                    self.averages_writer, self.config_map["averages_file"] = setup_csv_file(self.FLAGS.averages_file, self.get_averages_header(setup))
            if self.run_writer is None:
                self.run_writer = setup_run_file(self.FLAGS.run_file, header, self.config_map)
            if self.trajectory_writer is None:
                self.trajectory_writer = setup_trajectory_file(self.FLAGS.trajectory_file,
                                                               self.weights.get_total_dof(),
                                                               self.biases.get_total_dof(),
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
                                        "InfiniteSwitchSimulatedTempering"]:
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
                                    "InfiniteSwitchSimulatedTempering"]:
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
        return self.weights.get_total_dof()

    def get_total_bias_dof(self):
        return self.biases.get_total_dof()

    def sample(self, return_run_info = False, return_trajectories = False, return_averages = False):
        """ Performs the actual sampling of the neural network `nn` given a dataset `ds` and a
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
        # create global variable to hold kinetic energy
        with tf.variable_scope("accumulate", reuse=True):
            old_loss_t = tf.get_variable("old_loss", dtype=dds_basetype)
            old_kinetic_energy_t = tf.get_variable("old_kinetic", dtype=dds_basetype)
            kinetic_energy_t = tf.get_variable("kinetic", dtype=dds_basetype)
            zero_kinetic_energy = kinetic_energy_t.assign(0.)
            total_energy_t = tf.get_variable("total_energy", dtype=dds_basetype)
            momenta_t = tf.get_variable("momenta", dtype=dds_basetype)
            zero_momenta = momenta_t.assign(0.)
            gradients_t = tf.get_variable("gradients", dtype=dds_basetype)
            zero_gradients = gradients_t.assign(0.)
            virials_t = tf.get_variable("virials", dtype=dds_basetype)
            zero_virials = virials_t.assign(0.)
            noise_t = tf.get_variable("noise", dtype=dds_basetype)
            zero_noise = noise_t.assign(0.)
            accepted_t = tf.get_variable("accepted", dtype=tf.int64)
            zero_accepted = accepted_t.assign(0)
            rejected_t = tf.get_variable("rejected", dtype=tf.int64)
            zero_rejected = rejected_t.assign(0)

        placeholder_nodes = self.nn.get_dict_of_nodes(
            ["alpha", "friction_constant", "inverse_temperature", "inverse_temperature_max", "step_width", "current_step", "next_eval_step", "y_"])
        test_nodes = self.nn.get_list_of_nodes(["merged", "sample_step", "accuracy", "global_step", "loss", "y_", "y"])

        output_width = 8
        output_precision = 8

        written_row = 0

        accumulated_kinetic_energy = 0.
        accumulated_loss_nominator = 0.
        accumulated_loss_denominator = 0.
        accumulated_virials = 0.

        averages = None
        if return_averages:
            steps = (self.FLAGS.max_steps % self.FLAGS.every_nth)+1
            header = self.get_averages_header(setup="sample")
            no_params = len(header)
            averages = pd.DataFrame(
                np.zeros((steps, no_params)),
                columns=header)

        run_info = None
        if return_run_info:
            steps = (self.FLAGS.max_steps % self.FLAGS.every_nth)+1
            header = self.get_sample_header()
            no_params = len(header)
            run_info = pd.DataFrame(
                np.zeros((steps, no_params)),
                columns=header)

        trajectory = None
        if return_trajectories:
            steps = (self.FLAGS.max_steps % self.FLAGS.every_nth)+1
            header = get_trajectory_header(
                self.weights.get_total_dof(),
                self.biases.get_total_dof())
            no_params = self.weights.get_total_dof()+self.biases.get_total_dof()+3
            trajectory = pd.DataFrame(
                np.zeros((steps, no_params)),
                columns=header)

        # check that sampler's parameters are actually used
        if self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder",
                                  "GeometricLangevinAlgorithm_2ndOrder",
                                  "BAOAB"]:
            gamma, beta, deltat = self.sess.run(self.nn.get_list_of_nodes(
                ["friction_constant", "inverse_temperature", "step_width"]), feed_dict={
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["inverse_temperature"]: self.FLAGS.inverse_temperature,
                placeholder_nodes["friction_constant"]: self.FLAGS.friction_constant
            })
            logging.info("Sampler parameters: gamma = %lg, beta = %lg, delta t = %lg" %
                  (gamma, beta, deltat))
        elif self.FLAGS.sampler == "InfiniteSwitchSimulatedTempering":
            beta, beta_max, alpha, deltat = self.sess.run(self.nn.get_list_of_nodes(
                ["inverse_temperature", "inverse_temperature_max", "alpha", "step_width"]), feed_dict={
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["inverse_temperature"]: self.FLAGS.inverse_temperature,
                placeholder_nodes["inverse_temperature_max"]: self.FLAGS.inverse_temperature_max,
                placeholder_nodes["alpha"]: self.FLAGS.alpha,
            })
            logging.info("Sampler parameters: beta = %lg, beta_max = %lg, alpha = %lg, delta t = %lg" %
                         (beta, beta_max, alpha, deltat))
        elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
            current_step, num_mc_steps, deltat = self.sess.run(self.nn.get_list_of_nodes(
                ["current_step", "next_eval_step", "step_width"]), feed_dict={
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["current_step"]: 0,
                placeholder_nodes["next_eval_step"]: self.FLAGS.hamiltonian_dynamics_time
            })
            logging.info("Sampler parameters: current_step = %lg, num_mc_steps = %lg, delta t = %lg" %
                  (current_step, num_mc_steps, deltat))

        # create extra nodes for HMC
        if self.FLAGS.sampler in ["HamiltonianMonteCarlo", "InfiniteSwitchSimulatedTempering"]:
            HMC_eval_nodes = self.nn.get_list_of_nodes(["loss"]) + [total_energy_t, kinetic_energy_t]
            var_loss_t = tf.placeholder(old_loss_t.dtype.base_dtype, name="var_loss")
            var_kin_t = tf.placeholder(old_kinetic_energy_t.dtype.base_dtype, name="var_kinetic")
            var_total_t = tf.placeholder(total_energy_t.dtype.base_dtype, name="var_total")
            HMC_set_nodes = [old_loss_t.assign(var_loss_t),
                             old_kinetic_energy_t.assign(var_kin_t)]
            HMC_set_all_nodes = [total_energy_t.assign(var_total_t)]+HMC_set_nodes

        # zero rejection rate before sampling start
        check_accepted, check_rejected = self.sess.run([zero_accepted, zero_rejected])
        assert(check_accepted == 0)
        assert(check_rejected == 0)

        # prepare summaries for TensorBoard
        if self.FLAGS.summaries_path is not None:
            summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_path, self.sess.graph)

        logging.info("Starting to sample")
        logging.info_intervals = max(1, int(self.FLAGS.max_steps / 100))
        last_time = time.process_time()
        HMC_steps = 0
        for current_step in range(self.FLAGS.max_steps):
            # get next batch of data
            features, labels = self.input_pipeline.next_batch(self.sess)

            # pick next evaluation step with a little random variation
            if self.FLAGS.sampler == "HamiltonianMonteCarlo" and current_step > HMC_steps:
                HMC_steps += max(1,
                                round((0.9 + np.random.uniform(low=0., high=0.2)) \
                                      * self.FLAGS.hamiltonian_dynamics_time/self.FLAGS.step_width))
                logging.debug("Next evaluation of HMC criterion at step "+str(HMC_steps))

            # place in feed dict
            feed_dict = {
                self.xinput: features,
                placeholder_nodes["y_"]: labels,
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["inverse_temperature"]: self.FLAGS.inverse_temperature,
                placeholder_nodes["friction_constant"]: self.FLAGS.friction_constant,
                placeholder_nodes["current_step"]: current_step,
                placeholder_nodes["next_eval_step"]: HMC_steps,
                placeholder_nodes["inverse_temperature"]: self.FLAGS.inverse_temperature,
                placeholder_nodes["inverse_temperature_max"]: self.FLAGS.inverse_temperature_max,
                placeholder_nodes["alpha"]: self.FLAGS.alpha,
            }
            if self.FLAGS.dropout is not None:
                feed_dict.update({placeholder_nodes["keep_prob"] : self.FLAGS.dropout})
            #logging.debug("batch is x: "+str(features[:])+", y: "+str(labels[:]))

            # set global variable used in HMC sampler for criterion to initial loss
            if self.FLAGS.sampler in ["HamiltonianMonteCarlo", "InfiniteSwitchSimulatedTempering"]:
                loss_eval, total_eval, kin_eval = self.sess.run(HMC_eval_nodes, feed_dict=feed_dict)
                HMC_set_dict = {
                    var_kin_t: kin_eval,
                    var_loss_t: loss_eval,
                    var_total_t: loss_eval+kin_eval
                }
                if abs(total_eval) < 1e-10:
                    self.sess.run(HMC_set_all_nodes, feed_dict=HMC_set_dict)
                else:
                    self.sess.run(HMC_set_nodes, feed_dict=HMC_set_dict)
                loss_eval, total_eval, kin_eval = self.sess.run(HMC_eval_nodes, feed_dict=feed_dict)
                logging.debug("#%d: loss is %lg, total is %lg, kinetic is %lg" % (current_step, loss_eval, total_eval, kin_eval))

            # zero kinetic energy
            if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                      "GeometricLangevinAlgorithm_1stOrder",
                                      "GeometricLangevinAlgorithm_2ndOrder",
                                      "HamiltonianMonteCarlo",
                                      "BAOAB",
                                      "InfiniteSwitchSimulatedTempering"]:
                check_total, check_kinetic, check_momenta, check_gradients, check_virials, check_noise = \
                    self.sess.run([total_energy_t, zero_kinetic_energy, zero_momenta, zero_gradients, zero_virials, zero_noise])
                assert (abs(check_kinetic) < 1e-10)
                assert (abs(check_momenta) < 1e-10)
                assert (abs(check_gradients) < 1e-10)
                assert (abs(check_virials) < 1e-10)
                assert (abs(check_noise) < 1e-10)

            # get the weights and biases as otherwise the loss won't match
            # tf first computes loss, then gradient, then performs variable update
            # hence, after the sample step, we would have updated variables but old loss
            if current_step % self.FLAGS.every_nth == 0:
                if self.config_map["do_write_trajectory_file"] or return_trajectories:
                    weights_eval = self.weights.evaluate(self.sess)
                    biases_eval = self.biases.evaluate(self.sess)
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
            summary, _, acc, global_step, loss_eval, y_true_eval, y_eval = \
                self.sess.run(test_nodes, feed_dict=feed_dict)

            if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                      "GeometricLangevinAlgorithm_1stOrder",
                                      "GeometricLangevinAlgorithm_2ndOrder",
                                      "HamiltonianMonteCarlo",
                                      "BAOAB",
                                      "InfiniteSwitchSimulatedTempering"]:
                if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                    gradients, virials, noise = \
                        self.sess.run([gradients_t, virials_t, noise_t])
                    accumulated_virials += virials
                elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
                    old_total_energy, kinetic_energy, momenta, gradients, virials = \
                        self.sess.run([total_energy_t, kinetic_energy_t, momenta_t, gradients_t, virials_t])
                    accumulated_kinetic_energy += kinetic_energy
                    accumulated_virials += virials
                else:
                    kinetic_energy, momenta, gradients, virials, noise = \
                        self.sess.run([kinetic_energy_t, momenta_t, gradients_t, virials_t, noise_t])
                    accumulated_kinetic_energy += kinetic_energy
                    accumulated_virials += virials
            accumulated_loss_nominator += loss_eval * exp(- self.FLAGS.inverse_temperature * loss_eval)
            accumulated_loss_denominator += exp(- self.FLAGS.inverse_temperature * loss_eval)

            if self.FLAGS.summaries_path is not None:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary_writer.add_run_metadata(run_metadata, 'step%d' % current_step)
                summary_writer.add_summary(summary, current_step)

            if current_step % self.FLAGS.every_nth == 0:
                current_time = time.process_time()
                time_elapsed_per_nth_step = current_time - last_time
                last_time = current_time
                logging.debug("Output at step #" + str(current_step) + ", time elapsed till last is " + str(time_elapsed_per_nth_step))

                if self.config_map["do_write_trajectory_file"] or return_trajectories:
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

                if self.config_map["do_write_averages_file"] or return_averages:
                    averages_line = [0, global_step, current_step] \
                                    + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                                        precision=output_precision)] \
                                    + ['{:{width}.{precision}e}'.format(accumulated_loss_nominator/accumulated_loss_denominator, width=output_width,
                                                                        precision=output_precision)]
                    if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                        averages_line += ['{:{width}.{precision}e}'.format(abs(0.5 * accumulated_virials) / (float(current_step + 1.)), width=output_width,
                                                                           precision=output_precision)]
                    else:
                        averages_line += ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                       precision=output_precision)
                                      for x in [accumulated_kinetic_energy / float(current_step + 1.),
                                                abs(0.5 * accumulated_virials) / (float(current_step + 1.))]]

                    if self.config_map["do_write_averages_file"]:
                        self.averages_writer.writerow(averages_line)
                    if return_averages:
                        averages.loc[written_row] = averages_line

                if self.config_map["do_write_run_file"] or return_run_info:
                    run_line  = []
                    if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                              "GeometricLangevinAlgorithm_1stOrder",
                                              "GeometricLangevinAlgorithm_2ndOrder",
                                              "HamiltonianMonteCarlo",
                                              "BAOAB",
                                              "InfiniteSwitchSimulatedTempering"]:
                        run_line = [0, global_step, current_step] + ['{:1.3f}'.format(acc)] \
                                   + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                                       precision=output_precision)] \
                                   + ['{:{width}.{precision}e}'.format(time_elapsed_per_nth_step, width=output_width,
                                                                       precision=output_precision)]
                        if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                            run_line += ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                          precision=output_precision)
                                         for x in [sqrt(gradients), abs(0.5*virials), sqrt(noise)]]
                        elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
                            accepted_eval, rejected_eval  = self.sess.run([accepted_t, rejected_t])
                            if (rejected_eval+accepted_eval) > 0:
                                rejection_rate = rejected_eval/(rejected_eval+accepted_eval)
                            else:
                                rejection_rate = 0
                            run_line += ['{:{width}.{precision}e}'.format(loss_eval + kinetic_energy,
                                                                          width=output_width,
                                                                          precision=output_precision)]\
                                       + ['{:{width}.{precision}e}'.format(old_total_energy,
                                                                           width=output_width,
                                                                           precision=output_precision)]\
                                       + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                           precision=output_precision)
                                          for x in [kinetic_energy, sqrt(momenta), sqrt(gradients), abs(0.5*virials)]]\
                                       + ['{:{width}.{precision}e}'.format(rejection_rate, width=output_width,
                                                                           precision=output_precision)]
                        else:
                            run_line += ['{:{width}.{precision}e}'.format(loss_eval + kinetic_energy,
                                                                          width=output_width,
                                                                          precision=output_precision)]\
                                       + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                           precision=output_precision)
                                          for x in [kinetic_energy, sqrt(momenta), sqrt(gradients), abs(0.5*virials), sqrt(noise)]]

                    if self.config_map["do_write_run_file"]:
                        self.run_writer.writerow(run_line)
                    if return_run_info:
                        run_info.loc[written_row] = run_line
                written_row+=1

            #if (i % logging.info_intervals) == 0:
                #logging.debug('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                #logging.debug('Loss at step %s: %s' % (i, loss_eval))
                #logging.debug('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
                #logging.debug('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
        logging.info("SAMPLED.")

        # close summaries file
        if self.FLAGS.summaries_path is not None:
            summary_writer.close()

        return run_info, trajectory, averages

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
        with tf.variable_scope("accumulate", reuse=True):
            gradients_t = tf.get_variable("gradients", dtype=dds_basetype)
            zero_gradients = gradients_t.assign(0.)
            virials_t = tf.get_variable("virials", dtype=dds_basetype)
            zero_virials = virials_t.assign(0.)

        placeholder_nodes = self.nn.get_dict_of_nodes(["learning_rate", "y_"])
        test_nodes = self.nn.get_list_of_nodes(["merged", "train_step", "accuracy", "global_step",
                                                "loss", "y_", "y"])+[gradients_t]

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
                self.weights.get_total_dof(),
                self.biases.get_total_dof())
            no_params = self.weights.get_total_dof()+self.biases.get_total_dof()+3
            trajectory = pd.DataFrame(
                np.zeros((steps, no_params)),
                columns=header)

        logging.info("Starting to train")
        last_time = time.process_time()
        for current_step in range(self.FLAGS.max_steps):
            logging.debug("Current step is " + str(current_step))

            # get next batch of data
            features, labels = self.input_pipeline.next_batch(self.sess)

            # place in feed dict
            feed_dict = {
                self.xinput: features,
                placeholder_nodes["y_"]: labels,
                placeholder_nodes["learning_rate"]: self.FLAGS.step_width
            }
            if self.FLAGS.dropout is not None:
                feed_dict.update({placeholder_nodes["keep_prob"] : self.FLAGS.dropout})
            #logging.debug("batch is x: "+str(features[:])+", y: "+str(labels[:]))

            # zero accumulated gradient
            check_gradients, check_virials = self.sess.run([zero_gradients, zero_virials])
            assert (abs(check_gradients) < 1e-10)
            assert (abs(check_virials) < 1e-10)

            # get the weights and biases as otherwise the loss won't match
            # tf first computes loss, then gradient, then performs variable update
            # hence, after the sample step, we would have updated variables but old loss
            if current_step % self.FLAGS.every_nth == 0:
                if self.config_map["do_write_trajectory_file"] or return_trajectories:
                    weights_eval = self.weights.evaluate(self.sess)
                    biases_eval = self.biases.evaluate(self.sess)

            summary, _, acc, global_step, loss_eval, y_true_eval, y_eval, scaled_grad = \
                self.sess.run(test_nodes, feed_dict=feed_dict)

            gradients, virials = self.sess.run([gradients_t, virials_t])
            accumulated_virials += virials

            if current_step % self.FLAGS.every_nth == 0:
                current_time = time.process_time()
                time_elapsed_per_nth_step = current_time - last_time
                last_time = current_time
                logging.debug("Output at step #" + str(current_step) + ", time elapsed till last is " + str(time_elapsed_per_nth_step))

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
                    averages_line = [0, global_step, current_step] \
                                    + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                                        precision=output_precision)] \
                                    + ['{:{width}.{precision}e}'.format(abs(0.5 * accumulated_virials) / (float(current_step + 1.)), width=output_width,
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

        return run_info, trajectory, averages

    def compute_optimal_stepwidth(self):
        placeholder_nodes = self.nn.get_dict_of_nodes(["learning_rate", "y_"])

        # get first batch of data
        self.reset_dataset()
        features, labels = self.input_pipeline.next_batch(self.sess)

        # place in feed dict
        feed_dict = {
            self.xinput: features,
            placeholder_nodes["y_"]: labels,
            placeholder_nodes["learning_rate"]: self.FLAGS.step_width
        }
        if self.FLAGS.dropout is not None:
            feed_dict.update({placeholder_nodes["keep_prob"] : self.FLAGS.dropout})

        hessian_eval = self.sess.run(self.hessians, feed_dict=feed_dict)
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

    def finish(self):
        """ Closes all open files and saves the model if desired
        """
        self.close_files()

        try:
            if self.FLAGS.save_model is not None:
                save_path = self.save_model(self.FLAGS.save_model.replace('.meta', ''))
                logging.info("Model saved in file: %s" % save_path)
        except AttributeError:
            pass

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
        elif "bias" in _name:
            other_collection = tf.get_collection_ref(tf.GraphKeys.BIASES)
        else:
            logging.warning("Unknown parameter category for "+str(_name) \
                            +"), removing only from trainables.")
        for i in range(len(trainable_collection)):
            if trainable_collection[i].name == _name:
                trainable_variable = trainable_collection[i]
                del trainable_collection[i]
                break
        for i in range(len(other_collection)):
            if other_collection[i].name == _name:
                variable = other_collection[i]
                del other_collection[i]
                break
        logging.info("Comparing %s and %s" % (trainable_variable, variable))
        if trainable_variable is variable:
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
            # look for tensor in already fixed variables
            variable = None
            for var in self.fixed_variables:
                if var.name == name:
                    variable = var
                    break
            # if not found, fix. Otherwise, simply add
            if variable is None:
                retvariable = self._fix_parameter(name)
                if retvariable is not None:
                    self.fixed_variables.append(retvariable)
                    retlist.append(retvariable)
            else:
                retlist.append(variable)
        return retlist

    def assign_current_step(self, step):
        # set step
        if ('global_step' in self.nn.summary_nodes.keys()):
            sample_step_placeholder = self.nn.get("step_placeholder")
            feed_dict = {sample_step_placeholder: step}
            set_step = self.sess.run(self.global_step_assign_t, feed_dict=feed_dict)

    def assign_neural_network_parameters(self, parameters):
        """ Assigns the parameters of the neural network from
        the given array.

        :param parameters: list of values, one for each weight and bias
        """
        weights_dof = self.weights.get_total_dof()
        self.weights.assign(self.sess, parameters[0:weights_dof])
        self.biases.assign(self.sess, parameters[weights_dof:])

    def assign_weights_and_biases(self, weights_vals, biases_vals, do_check=False):
        """ Assigns weights and biases of a neural network.

        :param weights_vals: flat weights parameters
        :param biases_vals: flat bias parameters
        :param do_check: whether to check set values (and print) or not
        :return evaluated weights and bias on do_check or None otherwise
        """
        self.weights.assign(self.sess, weights_vals)
        self.biases.assign(self.sess, biases_vals)

        # get the input and biases to check against what we set
        if do_check:
            weights_eval = self.weights.evaluate(self.sess)
            biases_eval = self.biases.evaluate(self.sess)
            logging.info("Evaluating at weights " + str(weights_eval[0:10]) + ", biases " + str(biases_eval[0:10]))
            return weights_eval, biases_eval
        return None

    def assign_weights_and_biases_from_dataframe(self, df_parameters, rownr, do_check=False):
        """ Parse weight and bias values from a dataframe given a specific step
        to set the neural network's parameters.

        :param df_parameters: pandas dataframe
        :param rownr: rownr to set
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
        logging.info("Read row " + str(rownr) + ":" + str(parameters))

        # create internal array to store parameters
        weights_vals = self.weights.create_flat_vector()
        biases_vals = self.biases.create_flat_vector()
        weights_vals[:weights_vals.size] = [parameters[key] for key in sorted(parameters.keys()) if "w" in key]
        biases_vals[:biases_vals.size] = [parameters[key] for key in sorted(parameters.keys()) if "b" in key]
        return self.assign_weights_and_biases(weights_vals, biases_vals, do_check)

    def assign_weights_and_biases_from_file(self, filename, step, do_check=False):
        """ Parse weight and bias values from a CSV file given a specific step
        to set the neural network's parameters.

        :param filename: filename to parse
        :param step: step to set (i.e. value in "step" column designates row)
        :param do_check: whether to evaluate (and print) set parameters
        :return evaluated weights and bias on do_check or None otherwise
        """
        # parse csv file
        df_parameters = pd.read_csv(filename, sep=',', header=0)
        if step in df_parameters.loc[:, ['step']].values:
            rownr = np.where(df_parameters.loc[:, ['step']].values == step)[0]
            self.assign_current_step(step)
            return self.assign_weights_and_biases_from_dataframe(
                df_parameters=df_parameters,
                rownr=rownr,
                do_check=do_check
            )
        else:
            logging.debug("Step " + str(step) + " not found in file.")
            return None

    def assign_parameters(self, variables, values):
        """ Allows to assign multiple parameters at once.

        :param variables: list of tensorflow variables
        :param values: list of values to assign to
        """
        assert( len(variables) == len(values) )
        assigns=[]
        for i in range(len(variables)):
            value_t, assign_t = self._assign_parameter(
                variables[i],
                np.reshape(values[i], newshape=variables[i].shape))
            assigns.append(assign_t)
        self.sess.run(assigns)

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
