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
from builtins import staticmethod

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import variables

from TATi.common import create_input_layer, get_list_from_string, \
    initialize_config_map
from TATi.models.helpers import check_column_names_in_order, \
    get_start_index_in_dataframe_columns, \
    get_weight_and_bias_column_numbers
from TATi.models.input.inputpipelinefactory import InputPipelineFactory
from TATi.models.neuralnetwork import NeuralNetwork
from TATi.models.parameters.helpers import assign_parameter, fix_parameter, \
    fix_parameter_in_collection, split_collection_per_walker
from TATi.models.parameters.neuralnet_parameters import neuralnet_parameters
from TATi.models.trajectories.trajectory_sampling_factory import TrajectorySamplingFactory
from TATi.models.trajectories.trajectory_training import TrajectoryTraining
from TATi.models.trajectories.trajectorystate import TrajectoryState


class ModelState:
    """ Captures all of the state of the neural network model.

    This contains all ingredients such as input pipeline, network, trajectory,
    and so on. It is abstracted away from the actual interface to make the
    interface class as light-weight and simple to understand as possible.

    """
    def __init__(self, FLAGS):
        self.reset()
        self.reset_parameters(FLAGS)

    def reset(self):
        # for allowing just reusing a new instance of this class, make sure
        # that we also reset the default graph before we start setting up
        # the neural network
        tf.reset_default_graph()

        # reset options dict
        self.FLAGS = None

        # reset trajectory instances
        self.trajectorystate = None
        self.trajectory_sample = None
        self.trajectory_train = None

        self.number_of_parameters = 0  # number of biases and weights

        # mark input layer as to be created
        self.xinput = None
        self.x = None

        # mark already fixes variables
        self.fixed_variables = None

        # mark neuralnetwork, saver and session objects as to be created
        self.input_pipeline = None
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
        if self.trajectorystate is not None:
            self.trajectorystate.FLAGS = FLAGS

    def reset_dataset(self):
        """ Re-initializes the dataset for a new run
        """
        if self.input_pipeline is not None:
            self.input_pipeline.reset(self.sess)

    def provide_data(self, features, labels, shuffle=False):
        """ Use to provide an in-memory dataset, i.e., numpy arrays with
        `features` and `labels`.

        :param features: feature part of dataset
        :param labels: label part of dataset
        :param shuffle: whether to shuffle the dataset initially or not
        """
        self.input_pipeline = InputPipelineFactory.provide_data(
            self.FLAGS, features=features, labels=labels, shuffle=shuffle)

    def init_input_pipeline(self):
        InputPipelineFactory.scan_dataset_dimension_from_files(self.FLAGS)
        self.input_pipeline = InputPipelineFactory.create(self.FLAGS)
        #self.input_pipeline.reset(self.sess)

    def init_input_layer(self):
        # create input layer
        if self.xinput is None or self.x is None:
            input_columns = get_list_from_string(self.FLAGS.input_columns)
            self.xinput, self.x = create_input_layer(self.FLAGS.input_dimension, input_columns)

    def init_neural_network(self):
        if self.nn is None:
            self.nn = []
            self.loss = []
            self.trainables = []
            self.true_labels = NeuralNetwork.add_true_labels(self.FLAGS.output_dimension)

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
                        self.x, self.FLAGS.hidden_dimension, self.FLAGS.output_dimension,
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
                        self.FLAGS.output_type)
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
                            removed_vars = fix_parameter_in_collection(trainables, var, name_scope+"'s trainables")
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
        split_weights = split_collection_per_walker(
            tf.get_collection_ref(tf.GraphKeys.WEIGHTS), self.FLAGS.number_walkers)
        split_biases = split_collection_per_walker(
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
                assert( self.weights[i].get_total_dof() == self.weights[0].get_total_dof() )

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
                assert (self.biases[i].get_total_dof() == self.biases[0].get_total_dof())

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

    def assign_parse_parameter_file(self):
        # assign parameters of NN from step in given file
        if self.FLAGS.parse_parameters_file is not None \
                and (self.FLAGS.parse_steps is not None and (len(self.FLAGS.parse_steps) > 0)):
            step=self.FLAGS.parse_steps[0]
            for i in range(self.FLAGS.number_walkers):
                self.assign_weights_and_biases_from_file(self.FLAGS.parse_parameters_file, step,
                                                         walker_index=i, do_check=True)

    def init_trajectories(self):
        if self.trajectorystate is None:
            self.trajectorystate = TrajectoryState(self)
        if "sampler" in self.FLAGS and self.trajectory_sample is None:
            self.trajectory_sample = TrajectorySamplingFactory.create(
                self.FLAGS.sampler, self.trajectorystate)
        if "optimizer" in self.FLAGS and self.trajectory_train is None:
            self.trajectory_train = TrajectoryTraining(self.trajectorystate)

    def setup_trajectories(self, setup, prior):
        self.trajectorystate.init_trajectory(self)
        if setup is not None and "sample" in setup:
            self.trajectory_sample.init_trajectory(prior, self)
        if setup is not None and "train" in setup:
            self.trajectory_train.init_trajectory(prior, self)
        self.trajectorystate.init_step_placeholder()
        self.trajectorystate.init_parse_directions()

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
        assert (self.FLAGS.input_dimension is not None)

        self.init_input_layer()
        self.init_neural_network()

        fixed_variables, values = self.fix_variables()
        logging.debug("Remaining global trainable variables: " + str(variables.trainable_variables()))

        all_vectorized_gradients = self.init_vectorized_gradients(add_vectorized_gradients)
        self.init_hessians(all_vectorized_gradients)

        split_weights, split_biases = self.get_split_weights_and_biases()

        prior = self.init_prior()
        self.init_trajectories()
        self.setup_trajectories(setup, prior)

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

    def save_model(self, filename):
        """ Saves the current neural network model to a set of files,
        whose prefix is given by filename.

        See also `model.restore_model()`.

        :param filename: prefix of set of model files
        :return: path where model was saved
        """
        print("Saving model in" + filename)
        return self.saver.save(self.sess, filename)

    def restore_model(self, filename):
        """ Restores the model from a tensorflow checkpoint file.

        Compare to `model.save_model()`.

        :param filename: prefix of set of model files
        """
        # assign state of model from file if given
        if filename is not None:
            # Tensorflow DOCU says: initializing is not needed when restoring
            # however, global_variables are missing otherwise for storing kinetic, ...
            # tf.reset_default_graph()

            restore_path = filename.replace('.meta', '')
            self.saver.restore(self.sess, restore_path)
            logging.info("Model restored from file: %s" % restore_path)

    def finish(self):
        """ Closes all open files and saves the model if desired
        """
        try:
            if self.FLAGS.save_model is not None:
                save_path = self.save_model(self.FLAGS.save_model.replace('.meta', ''))
                logging.debug("Model saved in file: %s" % save_path)
        except AttributeError:
            pass

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
            retvariable_list = fix_parameter(name)
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
            value_t, assign_t = assign_parameter(
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

    def execute_trajectory_run(self, trajectory, return_run_info = False, return_trajectories = False, return_averages=False):
        retvals = trajectory.execute(self.sess,
            { "input_pipeline": self.input_pipeline,
              "xinput": self.xinput,
              "true_labels": self.true_labels},
            return_run_info, return_trajectories, return_averages)
        self.finish()
        return retvals


    def assign_current_step(self, step, walker_index=0):
        """ Allows to set the current step number of the iteration.

        :param step: step number to set
        :param walker_index: walker for which to set step
        """
        assert(walker_index < self.FLAGS.number_walkers)
        # set step
        if 'global_step' in self.nn[walker_index].summary_nodes.keys():
            sample_step_placeholder = self.trajectorystate.step_placeholder[walker_index]
            feed_dict = {sample_step_placeholder: step}
            set_step = self.sess.run(
                self.trajectorystate.global_step_assign_t[walker_index],
                feed_dict=feed_dict)
            assert (set_step == step)

    def assign_neural_network_parameters(self, parameters, walker_index=0, do_check=False):
        """ Assigns the parameters of the neural network from
        the given array.

        :param parameters: list of values, one for each weight and bias
        :param walker_index: index of the replicated network (in the graph)
        :param do_check: whether to check set values (and print) or not
        :return evaluated weights and bias on do_check or None otherwise
        """
        weights_dof = self.weights[walker_index].get_total_dof()
        return self.assign_weights_and_biases(weights_vals=parameters[0:weights_dof],
                                              biases_vals=parameters[weights_dof:],
                                              walker_index=walker_index, do_check=do_check)

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
            logging.info("Evaluating walker #"+str(walker_index)
                         + " at weights " + str(weights_eval[0:10])
                         + ", biases " + str(biases_eval[0:10]))
            assert(np.allclose(weights_eval, weights_vals, atol=1e-7))
            assert(np.allclose(biases_eval, biases_vals, atol=1e-7))
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
        weight_numbers, bias_numbers = get_weight_and_bias_column_numbers(df_parameters)
        values_aligned, _, _ = check_column_names_in_order(df_parameters,
                                                     weight_numbers, bias_numbers)

        if values_aligned:
            weights_start = get_start_index_in_dataframe_columns(
                weight_numbers, df_parameters, ["weight", "w"])
            biases_start = get_start_index_in_dataframe_columns(
                bias_numbers, df_parameters, ["bias", "b"])
            # copy values in one go
            weights_vals = df_parameters.iloc[rownr, weights_start:biases_start].values
            biases_vals = df_parameters.iloc[rownr, biases_start:].values
        else:
            # singly pick each value

            # create internal array to store parameters
            weights_vals = self.weights[walker_index].create_flat_vector()
            biases_vals = self.biases[walker_index].create_flat_vector()
            for keyname in df_parameters.columns:
                if "0" <= keyname[1] <= "9":
                    if "w" == keyname[0]:
                        weights_vals[int(keyname[1:])] = df_parameters.loc[rownr, [keyname]].values[0]
                    elif "b" == keyname[0]:
                        biases_vals[int(keyname[1:])] = df_parameters.loc[rownr, [keyname]].values[0]
                    else:
                        # not a parameter column
                        continue
                else:
                    if "weight" in keyname:
                        weights_vals[int(keyname[6:])] = df_parameters.loc[rownr, [keyname]].values[0]
                    elif "bias" in keyname:
                        biases_vals[int(keyname[4:])] = df_parameters.loc[rownr, [keyname]].values[0]

        logging.debug("Read row (first three weights and biases) " + str(rownr) + ":" + str(weights_vals[:5])
                      + "..." + str(biases_vals[:5]))

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
                num_ids = df_parameters.iloc[rowlist, id_idx].max() - df_parameters.iloc[rowlist, id_idx].min() + 1
                if num_ids >= self.FLAGS.number_walkers:
                    rowlist = np.where((df_parameters.iloc[rowlist, id_idx].values == walker_index))
                else:
                    logging.info("Not enough values in parse_parameters_file for all walkers, using first for all.")
            if len(rowlist) > 1:
                logging.warning("Found multiple matching entries to step " + str(step)
                                + " and walker #" + str(walker_index))
            elif len(rowlist) == 0:
                raise ValueError("Step " + str(step) + " and walker #" + str(walker_index)
                                 + " not found.")
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
