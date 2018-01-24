from builtins import staticmethod

import functools
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import time

from math import sqrt, floor, ceil

from DataDrivenSampler.common import create_input_layer, decode_csv_line, file_length, \
    get_csv_defaults, get_list_from_string, get_trajectory_header, \
    initialize_config_map, \
    setup_run_file, setup_trajectory_file
from DataDrivenSampler.models.input.datasetpipeline import DatasetPipeline
from DataDrivenSampler.models.mock_flags import MockFlags
from DataDrivenSampler.models.neuralnet_parameters import neuralnet_parameters
from DataDrivenSampler.models.neuralnetwork import NeuralNetwork


class model:
    """ This class combines the whole setup for creating a neural network.

    Moreover, it contains functions to either train or sample the loss function.

    """
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.config_map = initialize_config_map()

        # we train only on size batch and need as many epochs as tests
        self.FLAGS.dimension = sum([file_length(filename)
                                    for filename in FLAGS.batch_data_files]) \
                               - len(FLAGS.batch_data_files)
        print("Parsing "+str(FLAGS.batch_data_files))
        try:
            FLAGS.max_steps
        except AttributeError:
            FLAGS.max_steps = 1
        # self.batch_features, self.batch_labels = create_input_pipeline(
        #     FLAGS.batch_data_files,
        #     batch_size = batch_size,
        #     shuffle=False,
        #     num_epochs = max_steps,
        #     seed = FLAGS.seed)
        self.input_dimension = 2
        self.config_map["output_dimension"] = 1
        self.batch_next = self.create_input_pipeline(FLAGS)
        input_columns = get_list_from_string(FLAGS.input_columns)

        self.xinput, self.x = create_input_layer(self.input_dimension, input_columns)

        self.resources_created = None

        self.nn = None
        self.saver = None
        self.sess = None

        self.run_writer = None
        self.trajectory_writer = None

    def create_input_pipeline(self, FLAGS, shuffle=False):
        """ This creates an input pipeline using the tf.Dataset module.

        :param FLAGS: parameters
        :param shuffle: whether to shuffle dataset or not
        """
        self.input_pipeline = DatasetPipeline(filenames=FLAGS.batch_data_files,
                                              batch_size=FLAGS.batch_size, dimension=FLAGS.dimension, max_steps=FLAGS.max_steps,
                                              input_dimension=self.input_dimension, output_dimension=self.config_map["output_dimension"],
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
                                               use_resource=True, dtype=tf.float64)
            old_kinetic_t = tf.get_variable("old_kinetic", shape=[], trainable=False,
                                               initializer=tf.zeros_initializer,
                                               use_resource=True, dtype=tf.float64)
            total_energy_t = tf.get_variable("total_energy", shape=[], trainable=False,
                                               initializer=tf.zeros_initializer,
                                               use_resource=True, dtype=tf.float64)
            # used for Langevin samplers
            kinetic_energy_t = tf.get_variable("kinetic", shape=[], trainable=False,
                                               initializer=tf.zeros_initializer,
                                               use_resource=True, dtype=tf.float64)
            momenta_t = tf.get_variable("momenta", shape=[], trainable=False,
                                        initializer=tf.zeros_initializer,
                                        use_resource=True, dtype=tf.float64)
            gradients_t = tf.get_variable("gradients", shape=[], trainable=False,
                                          initializer=tf.zeros_initializer,
                                          use_resource=True, dtype=tf.float64)
            virials_t = tf.get_variable("virials", shape=[], trainable=False,
                                        initializer=tf.zeros_initializer,
                                        use_resource=True, dtype=tf.float64)
            noise_t = tf.get_variable("noise", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer,
                                      use_resource=True, dtype=tf.float64)
            # the following are used for HMC to measure rejection rate
            rejected_t = tf.get_variable("rejected", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer,
                                      use_resource=True, dtype=tf.int64)
            accepted_t = tf.get_variable("accepted", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer,
                                      use_resource=True, dtype=tf.int64)
        self.resources_created = True

    @staticmethod
    def setup_parameters(
            batch_data_files=[],
            batch_size=10,
            dropout=None,
            every_nth=1,
            fix_parameters=None,
            friction_constant=0.,
            hidden_activation="relu",
            hidden_dimension="",
            input_columns="1 2",
            inter_ops_threads=1,
            intra_ops_threads=None,
            inverse_temperature=1.,
            loss="mean_squared",
            max_steps=1000,
            hamiltonian_dynamics_steps=10,
            optimizer="GradientDescent",
            output_activation="tanh",
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
            trajectory_file=None):
            return MockFlags(
                batch_data_files=batch_data_files,
                batch_size=batch_size,
                dropout=dropout,
                every_nth=every_nth,
                fix_parameters=fix_parameters,
                friction_constant=friction_constant,
                hidden_activation=hidden_activation,
                hidden_dimension=hidden_dimension,
                input_columns=input_columns,
                inter_ops_threads=inter_ops_threads,
                intra_ops_threads=intra_ops_threads,
                inverse_temperature=inverse_temperature,
                loss=loss,
                max_steps=max_steps,
                hamiltonian_dynamics_steps=hamiltonian_dynamics_steps,
                optimizer=optimizer,
                output_activation=output_activation,
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
                trajectory_file=trajectory_file)

    def reset_dataset(self):
        """ Re-initializes the dataset for a new run
        """
        self.input_pipeline.reset(self.sess)

    def init_network(self, filename = None, setup = None):
        """ Initializes the graph, from a stored model if filename is not None.

        :param filename: name of file containing stored model
        """
        #if setup == "sample":
        self.create_resource_variables()

        if self.nn is None:
            self.nn = NeuralNetwork()
            hidden_dimension = get_list_from_string(self.FLAGS.hidden_dimension)
            activations = NeuralNetwork.get_activations()
            loss = self.nn.create(
                self.x, hidden_dimension, self.config_map["output_dimension"],
                seed=self.FLAGS.seed,
                add_dropped_layer=(self.FLAGS.dropout is not None),
                hidden_activation=activations[self.FLAGS.hidden_activation],
                output_activation=activations[self.FLAGS.output_activation],
                loss_name=self.FLAGS.loss
            )
        else:
            loss = self.nn.get_list_of_nodes(["loss"])[0]

        if self.FLAGS.fix_parameters is not None:
            names, values = self.split_parameters_as_names_values(self.FLAGS.fix_parameters)
            fixed_variables = self.fix_parameters(names)

        if setup == "train":
            self.nn.add_train_method(loss, optimizer_method=self.FLAGS.optimizer)
        elif setup == "sample":
            prior = {}
            if self.FLAGS.prior_factor is not None:
                prior["factor"] = self.FLAGS.prior_factor
            if self.FLAGS.prior_lower_boundary is not None:
                prior["lower_boundary"] = self.FLAGS.prior_lower_boundary
            if self.FLAGS.prior_power is not None:
                prior["power"] = self.FLAGS.prior_power
            if self.FLAGS.prior_upper_boundary is not None:
                prior["upper_boundary"] = self.FLAGS.prior_upper_boundary
            self.nn.add_sample_method(loss, sampling_method=self.FLAGS.sampler, seed=self.FLAGS.seed, prior=prior)
        else:
            print("Not adding sample or train method.")

        if self.saver is None:
            self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.WEIGHTS) +
                                   tf.get_collection(tf.GraphKeys.BIASES) + \
                                   tf.get_collection("Variables_to_Save"))
        if self.sess is None:
            #print("Using %s, %s threads " % (str(self.FLAGS.intra_ops_threads), str(self.FLAGS.inter_ops_threads)))
            self.sess = tf.Session(
                config=tf.ConfigProto(
                    intra_op_parallelism_threads=self.FLAGS.intra_ops_threads,
                    inter_op_parallelism_threads=self.FLAGS.inter_ops_threads))

        # initialize constants in graph
        self.nn.init_graph(self.sess)

        # initialize dataset
        self.input_pipeline.reset(self.sess)

        if self.FLAGS.fix_parameters is not None:
            self.assign_parameters(fixed_variables, values)

        if filename is not None:
            # Tensorflow DOCU says: initializing is not needed when restoring
            # however, global_variables are missing otherwise for storing kinetic, ...
            # tf.reset_default_graph()

            restore_path = filename.replace('.meta', '')
            self.saver.restore(self.sess, restore_path)
            print("Model restored from file: %s" % restore_path)

        header = None
        print("Setting up output files for "+str(setup))
        if setup == "sample":
            header = self.get_sample_header()
        elif setup == "train":
            header = self.get_train_header()

        self.weights = neuralnet_parameters(tf.get_collection(tf.GraphKeys.WEIGHTS))
        self.biases = neuralnet_parameters(tf.get_collection(tf.GraphKeys.BIASES))
        #print("There are %d weights and %d biases in the network"
        #      % (self.length_weights, self.length_biases))

        try:
            if self.run_writer is None:
                self.run_writer = setup_run_file(self.FLAGS.run_file, header, self.config_map)
            if self.trajectory_writer is None:
                self.trajectory_writer = setup_trajectory_file(self.FLAGS.trajectory_file,
                                                               self.weights.get_total_dof(),
                                                               self.biases.get_total_dof(),
                                                               self.config_map)
        except AttributeError:
            pass

    def get_sample_header(self):
        """ Prepares the distinct header for the run file for sampling
        """
        header = ['step', 'epoch', 'accuracy', 'loss', 'time_per_nth_step']
        if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
            header += ['scaled_gradient', 'virial', 'scaled_noise',
                      'average_virials']
        elif self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder",
                                    "GeometricLangevinAlgorithm_2ndOrder",
                                    "BAOAB"]:
            header += ['total_energy', 'kinetic_energy', 'scaled_momentum',
                      'scaled_gradient', 'virial', 'scaled_noise',
                      'average_kinetic_energy', 'average_virials']
        elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
            header += ['total_energy', 'old_total_energy', 'kinetic_energy', 'scaled_momentum',
                      'scaled_gradient', 'virial', 'average_kinetic_energy', 'average_virials',
                      'average_rejection_rate']
        return header

    def get_train_header(self):
        """ Prepares the distinct header for the run file for training
        """
        return ['step', 'epoch', 'accuracy', 'loss', 'time_per_nth_step', 'scaled_gradient', 'virial', 'average_virials']

    def sample(self, return_run_info = False, return_trajectories = False):
        """ Performs the actual sampling of the neural network `nn` given a dataset `ds` and a
        Session `session`.

        :param return_run_info: if set to true, run information will be accumulated
                inside a numpy array and returned
        :param return_trajectories: if set to true, trajectories will be accumulated
                inside a numpy array and returned (adhering to FLAGS.every_nth)
        :return: either twice None or a pandas dataframe depending on whether either
                parameter has evaluated to True
        """
        # create global variable to hold kinetic energy
        with tf.variable_scope("accumulate", reuse=True):
            old_loss_t = tf.get_variable("old_loss", dtype=tf.float64)
            old_kinetic_energy_t = tf.get_variable("old_kinetic", dtype=tf.float64)
            kinetic_energy_t = tf.get_variable("kinetic", dtype=tf.float64)
            zero_kinetic_energy = kinetic_energy_t.assign(0.)
            total_energy_t = tf.get_variable("total_energy", dtype=tf.float64)
            momenta_t = tf.get_variable("momenta", dtype=tf.float64)
            zero_momenta = momenta_t.assign(0.)
            gradients_t = tf.get_variable("gradients", dtype=tf.float64)
            zero_gradients = gradients_t.assign(0.)
            virials_t = tf.get_variable("virials", dtype=tf.float64)
            zero_virials = virials_t.assign(0.)
            noise_t = tf.get_variable("noise", dtype=tf.float64)
            zero_noise = noise_t.assign(0.)
            accepted_t = tf.get_variable("accepted", dtype=tf.int64)
            zero_accepted = accepted_t.assign(0)
            rejected_t = tf.get_variable("rejected", dtype=tf.int64)
            zero_rejected = rejected_t.assign(0)

        placeholder_nodes = self.nn.get_dict_of_nodes(
            ["friction_constant", "inverse_temperature", "step_width", "current_step", "num_steps", "y_"])
        test_nodes = self.nn.get_list_of_nodes(["merged", "sample_step", "accuracy", "global_step", "loss", "y_", "y"])

        output_width = 8
        output_precision = 8

        written_row = 0

        accumulated_kinetic_energy = 0.
        accumulated_virials = 0.

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
            no_params = self.weights.get_total_dof()+self.biases.get_total_dof()+2
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
        elif self.FLAGS.sampler == "HamiltonianMonteCarlo":
            current_step, num_mc_steps, deltat = self.sess.run(self.nn.get_list_of_nodes(
                ["current_step", "num_steps", "step_width"]), feed_dict={
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["current_step"]: 0,
                placeholder_nodes["num_steps"]: self.FLAGS.hamiltonian_dynamics_steps
            })
            print("Sampler parameters: current_step = %lg, num_mc_steps = %lg, delta t = %lg" %
                  (current_step, num_mc_steps, deltat))

        # create extra nodes for HMC
        if self.FLAGS.sampler == "HamiltonianMonteCarlo":
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

        print("Starting to sample")
        print_intervals = max(1, int(self.FLAGS.max_steps / 100))
        last_time = time.process_time()
        for i in range(self.FLAGS.max_steps):
            # get next batch of data
            features, labels = self.input_pipeline.next_batch(self.sess)

            # place in feed dict
            feed_dict = {
                self.xinput: features,
                placeholder_nodes["y_"]: labels,
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["inverse_temperature"]: self.FLAGS.inverse_temperature,
                placeholder_nodes["friction_constant"]: self.FLAGS.friction_constant,
                placeholder_nodes["current_step"]: i,
                placeholder_nodes["num_steps"]: self.FLAGS.hamiltonian_dynamics_steps
            }
            if self.FLAGS.dropout is not None:
                feed_dict.update({placeholder_nodes["keep_prob"] : self.FLAGS.dropout})
            # print("Testset is x: "+str(test_xs[:])+", y: "+str(test_ys[:]))
            # print("Testset is x: "+str(test_xs[:])+", y: "+str(test_ys[:]))

            # set global variable used in HMC sampler for criterion to initial loss
            if self.FLAGS.sampler == "HamiltonianMonteCarlo":
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
                #print("#%d: loss is %lg, total is %lg, kinetic is %lg" % (i, loss_eval, total_eval, kin_eval))

            # zero kinetic energy
            if self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder",
                                      "GeometricLangevinAlgorithm_2ndOrder",
                                      "HamiltonianMonteCarlo",
                                      "BAOAB"]:
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
            if i % self.FLAGS.every_nth == 0:
                if self.config_map["do_write_trajectory_file"] or return_trajectories:
                    weights_eval = self.weights.evaluate(self.sess)
                    biases_eval = self.biases.evaluate(self.sess)
                    #[print(str(item)) for item in weights_eval]
                    #[print(str(item)) for item in biases_eval]

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
                                      "BAOAB"]:
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
            if i % self.FLAGS.every_nth == 0:
                current_time = time.process_time()
                time_elapsed_per_nth_step = current_time - last_time
                last_time = current_time
                #print("Output at step #" + str(i) + ", time elapsed till last is " + str(time_elapsed_per_nth_step))

                if self.config_map["do_write_trajectory_file"] or return_trajectories:
                    trajectory_line = [global_step] \
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

                if self.config_map["do_write_run_file"] or return_run_info:
                    run_line  = []
                    if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                              "GeometricLangevinAlgorithm_1stOrder",
                                              "GeometricLangevinAlgorithm_2ndOrder",
                                              "HamiltonianMonteCarlo",
                                              "BAOAB"]:
                        run_line = [global_step, i] + ['{:1.3f}'.format(acc)] \
                                   + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                                       precision=output_precision)] \
                                   + ['{:{width}.{precision}e}'.format(time_elapsed_per_nth_step, width=output_width,
                                                                       precision=output_precision)]
                        if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                            run_line += ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                          precision=output_precision)
                                         for x in [sqrt(gradients), abs(0.5*virials), sqrt(noise), abs(0.5*accumulated_virials)/float(i+1.)]]
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
                                          for x in [kinetic_energy, sqrt(momenta), sqrt(gradients), abs(0.5*virials),
                                                    accumulated_kinetic_energy/float(i+1.), abs(0.5*accumulated_virials)/(float(i+1.))]]\
                                       + ['{:{width}.{precision}e}'.format(rejection_rate/(float(i)/self.FLAGS.hamiltonian_dynamics_steps+1.), width=output_width,
                                                                           precision=output_precision)]
                        else:
                            run_line += ['{:{width}.{precision}e}'.format(loss_eval + kinetic_energy,
                                                                          width=output_width,
                                                                          precision=output_precision)]\
                                       + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                           precision=output_precision)
                                          for x in [kinetic_energy, sqrt(momenta), sqrt(gradients), abs(0.5*virials), sqrt(noise),
                                                    accumulated_kinetic_energy/float(i+1.), abs(0.5*accumulated_virials)/float(i+1.)]]

                    if self.config_map["do_write_run_file"]:
                        self.run_writer.writerow(run_line)
                    if return_run_info:
                        run_info.loc[written_row] = run_line
                written_row+=1

            #if (i % print_intervals) == 0:
                #print('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                # print('Loss at step %s: %s' % (i, loss_eval))
                # print('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
                # print('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
        print("SAMPLED.")

        return run_info, trajectory

    def train(self, return_run_info = False, return_trajectories = False):
        """ Performs the actual training of the neural network `nn` given a dataset `ds` and a
        Session `session`.

        :param return_run_info: if set to true, run information will be accumulated
                inside a numpy array and returned
        :param return_trajectories: if set to true, trajectories will be accumulated
                inside a numpy array and returned (adhering to FLAGS.every_nth)
        :return: either twice None or a pandas dataframe depending on whether either
                parameter has evaluated to True
        """
        with tf.variable_scope("accumulate", reuse=True):
            gradients_t = tf.get_variable("gradients", dtype=tf.float64)
            zero_gradients = gradients_t.assign(0.)
            virials_t = tf.get_variable("virials", dtype=tf.float64)
            zero_virials = virials_t.assign(0.)

        placeholder_nodes = self.nn.get_dict_of_nodes(["step_width", "y_"])
        test_nodes = self.nn.get_list_of_nodes(["merged", "train_step", "accuracy", "global_step",
                                                "loss", "y_", "y"])+[gradients_t]

        output_width = 8
        output_precision = 8

        written_row = 0

        accumulated_virials = 0.

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
            no_params = self.weights.get_total_dof()+self.biases.get_total_dof()+2
            trajectory = pd.DataFrame(
                np.zeros((steps, no_params)),
                columns=header)

        print("Starting to train")
        last_time = time.process_time()
        for i in range(self.FLAGS.max_steps):
            #print("Current step is " + str(i))

            # get next batch of data
            features, labels = self.input_pipeline.next_batch(self.sess)

            # place in feed dict
            feed_dict = {
                self.xinput: features,
                placeholder_nodes["y_"]: labels,
                placeholder_nodes["step_width"]: self.FLAGS.step_width
            }
            if self.FLAGS.dropout is not None:
                feed_dict.update({placeholder_nodes["keep_prob"] : self.FLAGS.dropout})

            # zero accumulated gradient
            check_gradients, check_virials = self.sess.run([zero_gradients, zero_virials])
            assert (abs(check_gradients) < 1e-10)
            assert (abs(check_virials) < 1e-10)

            summary, _, acc, global_step, loss_eval, y_true_eval, y_eval, scaled_grad = \
                self.sess.run(test_nodes, feed_dict=feed_dict)

            gradients, virials = self.sess.run([gradients_t, virials_t])
            accumulated_virials += virials

            if i % self.FLAGS.every_nth == 0:
                current_time = time.process_time()
                time_elapsed_per_nth_step = current_time - last_time
                last_time = current_time
                #print("Output at step #" + str(i) + ", time elapsed till last is " + str(time_elapsed_per_nth_step))

                run_line = [global_step, i] + ['{:1.3f}'.format(acc)] \
                           + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                               precision=output_precision)] \
                           + ['{:{width}.{precision}e}'.format(time_elapsed_per_nth_step, width=output_width,
                                                               precision=output_precision)] \
                           + ['{:{width}.{precision}e}'.format(sqrt(gradients), width=output_width,
                                                               precision=output_precision)] \
                           + ['{:{width}.{precision}e}'.format(abs(0.5*virials),width=output_width,
                                                               precision=output_precision)] \
                           + ['{:{width}.{precision}e}'.format(abs(0.5*accumulated_virials)/float(i+1.),
                                                               width=output_width,
                                                               precision=output_precision)]
                if self.config_map["do_write_run_file"]:
                    self.run_writer.writerow(run_line)
                if return_run_info:
                    run_info.loc[written_row] = run_line
                if return_trajectories or self.config_map["do_write_trajectory_file"]:
                    weights_eval = self.weights.evaluate(self.sess)
                    biases_eval = self.biases.evaluate(self.sess)
                    trajectory_line = [global_step] \
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

            #print('Accuracy at step %s (%s): %s' % (i, global_step, acc))
            # print('Loss at step %s: %s' % (i, loss_eval))
            # print('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
            # print('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
        print("TRAINED down to loss %s and accuracy %s." % (loss_eval, acc))

        return run_info, trajectory

    def close_files(self):
        """ Closes the output files if they have been opened.
        """
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

    def finish(self):
        """ Closes all open files and saves the model if desired
        """
        self.close_files()

        try:
            if self.FLAGS.save_model is not None:
                save_path = self.saver.save(self.sess, self.FLAGS.save_model.replace('.meta', ''))
                print("Model saved in file: %s" % save_path)
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
            print("Unknown parameter category, removing only from trainables.")
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
        print("Comparing %s and %s" % (trainable_variable, variable))
        assert(trainable_variable is variable)
        return variable

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
        return [self._fix_parameter(name) for name in names]

    def assign_parameters(self, variables, values):
        """ Allows to assign multiple parameters at once.

        :param variables: list of tensorflow variables
        :param values: list of values to assign to
        """
        assert( len(variables) == len(values) )
        assigns=[]
        for i in range(len(variables)):
            value_t, assign_t = self._assign_parameter(variables[i],
                                                       np.reshape(values[i],
                                                           newshape=variables[i].shape,
                                                           ))
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
            b=a.split("=", 2)
            names.append(b[0])
            values.append(np.fromstring(b[1], dtype=float, sep=","))
        return names, values
