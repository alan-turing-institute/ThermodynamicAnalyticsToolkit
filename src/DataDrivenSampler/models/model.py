from builtins import staticmethod

import tensorflow as tf
import numpy as np
import sys
import pandas as pd

from math import sqrt, floor

from DataDrivenSampler.common import create_input_layer, file_length, \
    get_filename_from_fullpath, get_list_from_string, get_trajectory_header, \
    initialize_config_map, create_input_pipeline, setup_run_file, setup_trajectory_file
from DataDrivenSampler.datasets.classificationdatasets import ClassificationDatasets
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
        print("Parsing "+str(FLAGS.batch_data_files))
        self.FLAGS.dimension = sum([file_length(filename)
                                    for filename in FLAGS.batch_data_files]) \
                               - len(FLAGS.batch_data_files)
        self.batch_features, self.batch_labels = create_input_pipeline(
            FLAGS.batch_data_files,
            batch_size = FLAGS.batch_size,
            num_epochs = FLAGS.max_steps,
            seed = FLAGS.seed)
        input_dimension = 2
        self.config_map["output_dimension"] = 1
        input_columns = get_list_from_string(FLAGS.input_columns)

        self.xinput, self.x = create_input_layer(input_dimension, input_columns)

        self.resources_created = None

        self.nn = None
        self.saver = None
        self.sess = None

        print("weight vars: " + str(tf.get_collection(tf.GraphKeys.WEIGHTS)))
        print("bias vars: " + str(tf.get_collection(tf.GraphKeys.BIASES)))

        self.run_writer = None
        self.trajectory_writer = None

    def reset_flags(self, FLAGS):
        """ Use to pass a different set of FLAGS controlling training or sampling.

        :param FLAGS: new set of parameters
        """
        self.FLAGS = FLAGS

    def create_resource_variables(self):
        """ Creates some global resource variables to hold statistical quantities
        during sampling.
        """
        with tf.variable_scope("accumulate", reuse=self.resources_created):
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
        self.resources_created = True

    @staticmethod
    def create_mock_flags(
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
            optimizer="GradientDescent",
            output_activation="tanh",
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
                optimizer=optimizer,
                output_activation=output_activation,
                restore_model=restore_model,
                run_file=run_file,
                sampler=sampler,
                save_model=save_model,
                seed=seed,
                step_width=step_width,
                trajectory_file=trajectory_file)

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
            self.nn.add_sample_method(loss, sampling_method=self.FLAGS.sampler, seed=self.FLAGS.seed)
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

        self.nn.init_graph(self.sess)

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
        if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
            header = ['step', 'epoch', 'accuracy', 'loss', 'scaled_gradient', 'virial', 'scaled_noise',
                      'average_virials']
        elif self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder", "GeometricLangevinAlgorithm_2ndOrder"]:
            header = ['step', 'epoch', 'accuracy', 'loss', 'total_energy', 'kinetic_energy', 'scaled_momentum',
                      'scaled_gradient', 'virial', 'scaled_noise',
                      'average_kinetic_energy', 'average_virials']
        else:
            header = ['step', 'epoch', 'accuracy', 'loss']
        return header

    def get_train_header(self):
        """ Prepares the distinct header for the run file for training
        """
        return ['step', 'epoch', 'accuracy', 'loss', 'scaled_gradient', 'virial', 'average_virials']

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
            kinetic_energy_t = tf.get_variable("kinetic", dtype=tf.float64)
            zero_kinetic_energy = kinetic_energy_t.assign(0.)
            momenta_t = tf.get_variable("momenta", dtype=tf.float64)
            zero_momenta = momenta_t.assign(0.)
            gradients_t = tf.get_variable("gradients", dtype=tf.float64)
            zero_gradients = gradients_t.assign(0.)
            virials_t = tf.get_variable("virials", dtype=tf.float64)
            zero_virials = virials_t.assign(0.)
            noise_t = tf.get_variable("noise", dtype=tf.float64)
            zero_noise = noise_t.assign(0.)

        placeholder_nodes = self.nn.get_dict_of_nodes(
            ["friction_constant", "inverse_temperature", "step_width", "y_"])
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
        if self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder", "GeometricLangevinAlgorithm_2ndOrder"]:
            gamma, beta, deltat = self.sess.run(self.nn.get_list_of_nodes(
                ["friction_constant", "inverse_temperature", "step_width"]), feed_dict={
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["inverse_temperature"]: self.FLAGS.inverse_temperature,
                placeholder_nodes["friction_constant"]: self.FLAGS.friction_constant
            })
            print("Sampler parameters: gamma = %lg, beta = %lg, delta t = %lg" % (gamma, beta, deltat))

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        print("Starting to sample")
        print_intervals = max(1, int(self.FLAGS.max_steps / 100))
        for i in range(self.FLAGS.max_steps):
            # print("Current step is "+str(i))

            # fetch next batch of data
            try:
                batch_xs, batch_ys = self.sess.run([self.batch_features, self.batch_labels])
            except tf.errors.OutOfRangeError:
                print('End of epoch reached too early!')
                sys.exit(255)

            # place in feed dict
            feed_dict = {
                self.xinput: batch_xs, placeholder_nodes["y_"]: batch_ys,
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["inverse_temperature"]: self.FLAGS.inverse_temperature,
                placeholder_nodes["friction_constant"]: self.FLAGS.friction_constant
            }
            if self.FLAGS.dropout is not None:
                feed_dict.update({placeholder_nodes["keep_prob"] : self.FLAGS.dropout})
            # print("Testset is x: "+str(test_xs[:])+", y: "+str(test_ys[:]))
            # print("Testset is x: "+str(test_xs[:])+", y: "+str(test_ys[:]))

            # zero kinetic energy
            if self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder", "GeometricLangevinAlgorithm_2ndOrder"]:
                check_kinetic, check_momenta, check_gradients, check_virials, check_noise = \
                    self.sess.run([zero_kinetic_energy, zero_momenta, zero_gradients, zero_virials, zero_noise])
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
                                      "GeometricLangevinAlgorithm_2ndOrder"]:
                if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                    gradients, virials, noise = \
                        self.sess.run([gradients_t, virials_t, noise_t])
                    accumulated_virials += virials
                else:
                    kinetic_energy, momenta, gradients, virials, noise = \
                        self.sess.run([kinetic_energy_t, momenta_t, gradients_t, virials_t, noise_t])
                    accumulated_kinetic_energy += kinetic_energy
                    accumulated_virials += virials
            if i % self.FLAGS.every_nth == 0:
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
                                              "GeometricLangevinAlgorithm_2ndOrder"]:
                        if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                            run_line = [global_step, i] + ['{:1.3f}'.format(acc)] \
                                       + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                                           precision=output_precision)] \
                                       + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                           precision=output_precision)
                                          for x in [sqrt(gradients), abs(0.5*virials), sqrt(noise), abs(0.5*accumulated_virials)/float(i+1.)]]
                        else:
                            run_line = [global_step, i] + ['{:1.3f}'.format(acc)] \
                                       + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                                           precision=output_precision)] \
                                       + ['{:{width}.{precision}e}'.format(loss_eval + kinetic_energy,
                                                                           width=output_width,
                                                                           precision=output_precision)]\
                                       + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                           precision=output_precision)
                                          for x in [kinetic_energy, sqrt(momenta), sqrt(gradients), abs(0.5*virials), sqrt(noise),
                                                    accumulated_kinetic_energy/float(i+1.), abs(0.5*accumulated_virials)/float(i+1.)]]
                    else:
                        run_line = [global_step, i] + ['{:1.3f}'.format(acc)] \
                                   + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
                                                                       precision=output_precision)]

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

        coord.request_stop()
        coord.join(threads)

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
            zero_virials = gradients_t.assign(0.)

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

        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

        print("Starting to train")
        for i in range(self.FLAGS.max_steps):
            print("Current step is " + str(i))

            # fetch next batch of data
            try:
                batch_xs, batch_ys = self.sess.run([self.batch_features, self.batch_labels])
            except tf.errors.OutOfRangeError:
                print('End of epoch reached too early!')
                sys.exit(255)

            # place in feed dict
            feed_dict = {
                self.xinput: batch_xs, placeholder_nodes["y_"]: batch_ys,
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
                run_line = [global_step, i] + ['{:1.3f}'.format(acc)] \
                           + ['{:{width}.{precision}e}'.format(loss_eval, width=output_width,
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
        coord.request_stop()
        coord.join(threads)

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
        collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i in range(len(collection)):
            if collection[i].name == _name:
                variable = collection[i]
                del collection[i]
                break
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
