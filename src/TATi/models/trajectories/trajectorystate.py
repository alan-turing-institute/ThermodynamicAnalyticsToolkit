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

from builtins import staticmethod
import logging
import numpy as np
import tensorflow as tf
import time

from TATi.models.basetype import dds_basetype

class TrajectoryState(object):
    """ TrajectoryState contains the *unique* state of all trajectories.

    `TrajectoryState` needs to be unique and is shared as ref among
    Trajectory classes through their base class `TrajectoryBase`.

    Here, we create all (unique) objects required to perform the evaluate
    a trajectory through training or sampling the loss.

    More generally, all functionality for adding of nodes to and evaluation
    of nodes in the computational graph in the course of extracting
    trajectories is contained here.

    """
    def __init__(self, model):
        # mark step assign op as to be created
        self.step_placeholder = None
        self.global_step_assign_t = None
        self.global_step = None

        self.summary = None

        # store all ref to options
        self.FLAGS = model.FLAGS

        # mark resource variables as to be created
        self.resources_created = None

        self.directions = None

    def _get_all_parameters(self):
        all_weights = []
        all_biases = []
        for walker_index in range(self.FLAGS.number_walkers):
            all_weights.append(self.weights[walker_index].parameters)
            all_biases.append(self.biases[walker_index].parameters)
        return all_weights, all_biases

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

    def init_step_placeholder(self):
        if self.step_placeholder is None:
            self.step_placeholder = []
            for i in range(self.FLAGS.number_walkers):
                with tf.name_scope("walker"+str(i+1)):
                    self.step_placeholder.append(tf.placeholder(shape=(), dtype=tf.int32))
        if self.global_step_assign_t is None:
            self.global_step_assign_t = []
            for i in range(self.FLAGS.number_walkers):
                global_step = self.nn[i]._prepare_global_step()
                with tf.name_scope("walker"+str(i+1)):
                    self.global_step_assign_t.append(tf.assign(global_step,
                                                               self.step_placeholder[i]))

    def _prepare_summaries(self, session):
        summary_writer = None
        if self.FLAGS.summaries_path is not None:
            summary_writer = tf.summary.FileWriter(self.FLAGS.summaries_path, session.graph)
        return summary_writer

    def _write_summaries(self, summary_writer, summary, current_step):
        if self.FLAGS.summaries_path is not None:
            summary_writer.add_run_metadata(summary[1], 'step%d' % current_step)
            summary_writer.add_summary(summary[0], current_step)

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
                        TrajectoryState._dict_append(static_var_dict, key, static_var)
                        zero_assigner = static_var.assign(0.)
                        TrajectoryState._dict_append(zero_assigner_dict, key, zero_assigner)
                    for key in static_vars_int64:
                        static_var = tf.get_variable(key, dtype=tf.int64)
                        TrajectoryState._dict_append(static_var_dict, key, static_var)
                        zero_assigner = static_var.assign(0)
                        TrajectoryState._dict_append(zero_assigner_dict, key, zero_assigner)

                    for key in ["current_kinetic", "old_total_energy"]:
                        placeholder = tf.placeholder(dds_basetype, name=key)
                        TrajectoryState._dict_append(placeholder_dict, key, placeholder)
                        assigner = static_var_dict[key][i].assign(placeholder)
                        TrajectoryState._dict_append(assigner_dict, key, assigner)

        return static_var_dict, zero_assigner_dict, assigner_dict, placeholder_dict

    def _zero_state_variables(self, session, method):
        if method in ["GradientDescent",
                      "StochasticGradientLangevinDynamics",
                      "GeometricLangevinAlgorithm_1stOrder",
                      "GeometricLangevinAlgorithm_2ndOrder",
                      "HamiltonianMonteCarlo_1stOrder",
                      "HamiltonianMonteCarlo_2ndOrder",
                      "BAOAB",
                      "CovarianceControlledAdaptiveLangevinThermostat"]:
            check_kinetic, check_inertia, check_momenta, check_gradients, check_virials, check_noise = \
                session.run([
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

    def init_trajectory(self, model):
        # create global variables, one for every walker in its replicated graph
        self.create_resource_variables()
        self.static_vars, self.zero_assigner, self.assigner, self.placeholders = \
            self._create_static_variable_dict(self.FLAGS.number_walkers)

        # store ref to nn and its bias and weight arrays
        self.nn = model.nn
        self.biases = model.biases
        self.weights = model.weights

        # store model specific refs
        self.xinput = model.xinput
        self.true_labels = model.true_labels

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

    def _perform_step(self, session, test_nodes, feed_dict):
        summary = None
        run_metadata = None
        if self.FLAGS.summaries_path is not None:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            results = session.run(test_nodes,
                                       feed_dict=feed_dict,
                                       options=run_options,
                                       run_metadata=run_metadata)
            summary, acc, global_step, loss_eval = \
                results[0], results[2], results[3], results[4]

        else:
            results = session.run(test_nodes,
                                       feed_dict=feed_dict)
            acc, global_step, loss_eval = \
                results[1], results[2], results[3]

        return [summary, run_metadata], acc, global_step, loss_eval


    def _decide_collapse_walkers(self, session, current_step):
        # collapse walkers' positions onto first walker if desired and after having
        # recomputed the covariance matrix
        if (self.FLAGS.number_walkers > 1) and (self.FLAGS.collapse_walkers) and \
                (current_step % self.FLAGS.covariance_after_steps == 0) and \
                (current_step != 0):
            #print("COLLAPSING " + str(self.FLAGS.collapse_walkers))
            # get walker 0's position
            weights_eval, biases_eval = session.run([
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
            session.run(assigns, feed_dict=collapse_feed_dict)

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
