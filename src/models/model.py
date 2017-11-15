import tensorflow as tf
import numpy as np
import sys

from math import sqrt
from DataDrivenSampler.common import create_classification_dataset, \
    construct_network_model, get_activations, get_filename_from_fullpath, \
    initialize_config_map, setup_run_file, setup_trajectory_file


class model:
    """ This class combines the whole setup for creating a neural network.

    Moreover, it contains functions to either train or sample the loss function.

    """
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.config_map = initialize_config_map()

        # init random: None will use random seed
        if FLAGS.seed is not None:
            np.random.seed(FLAGS.seed)

        self.xinput, self.x, self.ds = create_classification_dataset(self.FLAGS, self.config_map)

        self.nn = None
        self.saver = None
        self.sess = None

        #print("weight vars: " + str(tf.get_collection(tf.GraphKeys.WEIGHTS)))
        #print("bias vars: " + str(tf.get_collection(tf.GraphKeys.BIASES)))

        self.run_writer = None
        self.trajectory_writer = None

    @staticmethod
    def create_resource_variables():
        """ Creates some global resource variables to hold statistical quantities
        during sampling.
        """
        with tf.variable_scope("accumulate"):
            kinetic_energy_t = tf.get_variable("kinetic", shape=[], trainable=False,
                                               initializer=tf.zeros_initializer,
                                               use_resource=True)
            momenta_t = tf.get_variable("momenta", shape=[], trainable=False,
                                        initializer=tf.zeros_initializer,
                                        use_resource=True)
            gradients_t = tf.get_variable("gradients", shape=[], trainable=False,
                                          initializer=tf.zeros_initializer,
                                          use_resource=True)
            noise_t = tf.get_variable("noise", shape=[], trainable=False,
                                      initializer=tf.zeros_initializer,
                                      use_resource=True)

    def init_network(self, filename = None, setup = None):
        """ Initializes the graph, from a stored model if filename is not None.

        :param filename: name of file containing stored model
        """
        if setup == "sample":
            self.create_resource_variables()

        activations = get_activations()
        self.nn = construct_network_model(self.FLAGS, self.config_map, self.x,
                                          hidden_activation=activations[self.FLAGS.hidden_activation],
                                          output_activation=activations[self.FLAGS.output_activation],
                                          loss_name=self.FLAGS.loss)

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.WEIGHTS) +
                               tf.get_collection(tf.GraphKeys.BIASES) + \
                               tf.get_collection("Variables_to_Save"))
        self.sess = tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=None,
                inter_op_parallelism_threads=1))

        self.nn.init_graph(self.sess)

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
        else:
            print("Unknown setup desired for the model")
            sys.exit(255)
        self.run_writer = setup_run_file(self.FLAGS.run_file, header, self.config_map)
        self.trajectory_writer = setup_trajectory_file(self.FLAGS.trajectory_file,
                                                       self.nn.get("weights").get_shape()[0],
                                                       self.nn.get("biases").get_shape()[0],
                                                       self.config_map)

    def get_sample_header(self):
        """ Prepares the distinct header for the run file for sampling
        """
        if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
            header = ['step', 'epoch', 'accuracy', 'loss', 'scaled_gradient', 'scaled_noise']
        elif self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder", "GeometricLangevinAlgorithm_2ndOrder"]:
            header = ['step', 'epoch', 'accuracy', 'loss', 'total_energy', 'kinetic_energy', 'scaled_momentum',
                      'scaled_gradient', 'scaled_noise']
        else:
            header = ['step', 'epoch', 'accuracy', 'loss']
        return header

    def get_train_header(self):
        """ Prepares the distinct header for the run file for training
        """
        return ['step', 'epoch', 'accuracy', 'loss', 'scaled_gradient']

    def sample(self):
        """ Performs the actual sampling of the neural network `nn` given a dataset `ds` and a
        Session `session`.
        """
        # create global variable to hold kinetic energy
        with tf.variable_scope("accumulate", reuse=True):
            kinetic_energy_t = tf.get_variable("kinetic")
            zero_kinetic_energy = kinetic_energy_t.assign(0.)
            momenta_t = tf.get_variable("momenta")
            zero_momenta = momenta_t.assign(0.)
            gradients_t = tf.get_variable("gradients")
            zero_gradients = gradients_t.assign(0.)
            noise_t = tf.get_variable("noise")
            zero_noise = noise_t.assign(0.)

        placeholder_nodes = self.nn.get_dict_of_nodes(
            ["friction_constant", "inverse_temperature", "step_width", "y_"])
        test_nodes = self.nn.get_list_of_nodes(["merged", "train_step", "accuracy", "global_step", "loss", "y_", "y"])

        output_width = 8
        output_precision = 8

        # check that sampler's parameters are actually used
        if self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder", "GeometricLangevinAlgorithm_2ndOrder"]:
            gamma, beta, deltat = self.sess.run(self.nn.get_list_of_nodes(
                ["friction_constant", "inverse_temperature", "step_width"]), feed_dict={
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["inverse_temperature"]: self.FLAGS.inverse_temperature,
                placeholder_nodes["friction_constant"]: self.FLAGS.friction_constant
            })
            print("Sampler parameters: gamma = %lg, beta = %lg, delta t = %lg" % (gamma, beta, deltat))

        print("Starting to sample")
        print_intervals = max(1, int(self.FLAGS.max_steps / 100))
        for i in range(self.FLAGS.max_steps):
            # print("Current step is "+str(i))
            batch_xs, batch_ys = self.ds.next_batch(self.FLAGS.batch_size)
            feed_dict = {
                self.xinput: batch_xs, placeholder_nodes["y_"]: batch_ys,
                placeholder_nodes["step_width"]: self.FLAGS.step_width,
                placeholder_nodes["inverse_temperature"]: self.FLAGS.inverse_temperature,
                placeholder_nodes["friction_constant"]: self.FLAGS.friction_constant
            }
            # print("Testset is x: "+str(test_xs[:])+", y: "+str(test_ys[:]))
            # print("Testset is x: "+str(test_xs[:])+", y: "+str(test_ys[:]))

            # zero kinetic energy
            if self.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder", "GeometricLangevinAlgorithm_2ndOrder"]:
                check_kinetic, check_momenta, check_gradients, check_noise = \
                    self.sess.run([zero_kinetic_energy, zero_momenta, zero_gradients, zero_noise])
                assert (abs(check_kinetic) < 1e-10)
                assert (abs(check_momenta) < 1e-10)
                assert (abs(check_gradients) < 1e-10)
                assert (abs(check_noise) < 1e-10)
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
            if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics", "GeometricLangevinAlgorithm_1stOrder",
                                 "GeometricLangevinAlgorithm_2ndOrder"]:
                if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                    gradients, noise = \
                        self.sess.run([gradients_t, noise_t])
                else:
                    kinetic_energy, momenta, gradients, noise = \
                        self.sess.run([kinetic_energy_t, momenta_t, gradients_t, noise_t])
            if i % self.FLAGS.every_nth == 0:
                if self.config_map["do_write_trajectory_file"]:
                    weights_eval, biases_eval = self.sess.run(
                        [self.nn.get("weights"), self.nn.get("biases")],
                        feed_dict=feed_dict)
                    self.trajectory_writer.writerow(
                        [global_step, loss_eval]
                        + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                           for sublist in weights_eval for item in sublist]
                        + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                           for item in biases_eval])

                if self.config_map["do_write_run_file"]:
                    if self.FLAGS.sampler in ["StochasticGradientLangevinDynamics", "GeometricLangevinAlgorithm_1stOrder",
                                         "GeometricLangevinAlgorithm_2ndOrder"]:
                        if self.FLAGS.sampler == "StochasticGradientLangevinDynamics":
                            self.run_writer.writerow([global_step, i, acc, loss_eval]
                                                + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                                    precision=output_precision)
                                                   for x in [sqrt(gradients), sqrt(noise)]])
                        else:
                            self.run_writer.writerow([global_step, i, acc, loss_eval]
                                                + ['{:{width}.{precision}e}'.format(loss_eval + kinetic_energy,
                                                                                    width=output_width,
                                                                                    precision=output_precision)]
                                                + ['{:{width}.{precision}e}'.format(x, width=output_width,
                                                                                    precision=output_precision)
                                                   for x in
                                                   [kinetic_energy, sqrt(momenta), sqrt(gradients), sqrt(noise)]])
                    else:
                        self.run_writer.writerow([global_step, i, acc, loss_eval])

            if (i % print_intervals) == 0:
                print('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                # print('Loss at step %s: %s' % (i, loss_eval))
                # print('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
                # print('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
        print("SAMPLED.")

    def train(self):
        """ Performs the actual training of the neural network `nn` given a dataset `ds` and a
        Session `session`.
        """
        placeholder_nodes = self.nn.get_dict_of_nodes(["step_width", "y_"])
        test_nodes = self.nn.get_list_of_nodes(["merged", "train_step", "accuracy", "global_step",
                                           "loss", "y_", "y", "scaled_gradient"])
        output_width = 8
        output_precision = 8

        print("Starting to train")
        for i in range(self.FLAGS.max_steps):
            print("Current step is " + str(i))
            batch_xs, batch_ys = self.ds.next_batch(self.FLAGS.batch_size)
            feed_dict = {
                self.xinput: batch_xs, placeholder_nodes["y_"]: batch_ys,
                placeholder_nodes["step_width"]: self.FLAGS.step_width
            }
            summary, _, acc, global_step, loss_eval, y_true_eval, y_eval, scaled_grad = \
                self.sess.run(test_nodes, feed_dict=feed_dict)

            if i % self.FLAGS.every_nth == 0:
                if self.config_map["do_write_run_file"]:
                    self.run_writer.writerow([global_step, i, acc, loss_eval, scaled_grad])
                if self.config_map["do_write_trajectory_file"]:
                    weights_eval, biases_eval = \
                        self.sess.run([self.nn.get("weights"), self.nn.get("biases")], feed_dict=feed_dict)
                    self.trajectory_writer.writerow(
                        [global_step, loss_eval]
                        + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                           for sublist in weights_eval for item in sublist]
                        + ['{:{width}.{precision}e}'.format(item, width=output_width, precision=output_precision)
                           for item in biases_eval])

            print('Accuracy at step %s (%s): %s' % (i, global_step, acc))
            # print('Loss at step %s: %s' % (i, loss_eval))
            # print('y_ at step %s: %s' % (i, str(y_true_eval[0:9].transpose())))
            # print('y at step %s: %s' % (i, str(y_eval[0:9].transpose())))
        print("TRAINED.")

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

        if self.FLAGS.save_model is not None:
            save_path = self.saver.save(self.sess, self.FLAGS.save_model.replace('.meta', ''))
            print("Model saved in file: %s" % save_path)

