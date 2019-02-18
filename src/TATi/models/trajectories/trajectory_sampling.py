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
import tensorflow as tf

from TATi.models.trajectories.trajectory_base import TrajectoryBase

class TrajectorySampling(TrajectoryBase):
    """Refines the Trajectory class to perform a sampling trajectory."""
    def __init__(self, trajectory_state):
        super(TrajectorySampling, self).__init__(trajectory_state)
        self.sampler = None
        self.EQN_nodes = None

    def get_placeholder_nodes(self):
        self.EQN_nodes = [self.state.nn[walker_index].get("EQN_step") \
                          for walker_index in range(self.state.FLAGS.number_walkers)]
        return []

    def init_accumulator(self):
        # set inverse temperature in case of sampling (not training)
        self._init_accumulator(self.state.FLAGS.sampler)
        self.averages.inverse_temperature = self.state.FLAGS.inverse_temperature,

    def _print_parameters(self, session, feed_dict):
        for walker_index in range(self.state.FLAGS.number_walkers):
            logging.info("Dependent walker #"+str(walker_index))
            if self.state.FLAGS.covariance_blending != 0.:
                eta =  session.run(self.state.nn[walker_index].get_list_of_nodes(
                    ["covariance_blending"]), feed_dict=feed_dict)[0]
                logging.info("EQN parameters, walker #%d: eta = %lg" %
                             (walker_index, eta))
            if self.state.FLAGS.sampler in ["StochasticGradientLangevinDynamics",
                                      "GeometricLangevinAlgorithm_1stOrder",
                                      "GeometricLangevinAlgorithm_2ndOrder",
                                      "BAOAB",
                                      "CovarianceControlledAdaptiveLangevinThermostat"]:
                gamma, beta, deltat = session.run(self.state.nn[walker_index].get_list_of_nodes(
                    ["friction_constant", "inverse_temperature", "step_width"]), feed_dict=feed_dict)
                logging.info("LD Sampler parameters, walker #%d: gamma = %lg, beta = %lg, delta t = %lg" %
                      (walker_index, gamma, beta, deltat))
            elif "HamiltonianMonteCarlo" in self.state.FLAGS.sampler:
                current_step, num_mc_steps, hd_steps, deltat = session.run(self.state.nn[walker_index].get_list_of_nodes(
                    ["current_step", "next_eval_step", "hamiltonian_dynamics_steps", "step_width"]), feed_dict=feed_dict)
                logging.info("MC Sampler parameters, walker #%d: current_step = %lg, num_mc_steps = %lg, HD_steps = %lg, delta t = %lg" %
                      (walker_index, current_step, num_mc_steps, hd_steps, deltat))
            else:
                raise NotImplementedError("The sampler method %s is unknown" % (self.state.FLAGS.sampler))

    def get_averages_header(self):
        """Prepares the distinct header for the averages file for sampling

        Args:

        Returns:

        """
        header = super(TrajectorySampling, self).get_averages_header()
        if self.state.FLAGS.sampler == "StochasticGradientLangevinDynamics":
            header += ['ensemble_average_loss', 'average_virials']
        elif self.state.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder",
                                    "GeometricLangevinAlgorithm_2ndOrder",
                                    "BAOAB",
                                    "CovarianceControlledAdaptiveLangevinThermostat"]:
            header += ['ensemble_average_loss', 'average_kinetic_energy', 'average_virials', 'average_inertia']
        elif "HamiltonianMonteCarlo" in self.state.FLAGS.sampler:
            header += ['ensemble_average_loss', 'average_kinetic_energy', 'average_virials', 'average_inertia',
                       'average_rejection_rate']
        return header

    def get_run_header(self):
        """Prepares the distinct header for the run file for sampling"""
        header = ['id', 'step', 'epoch', 'accuracy', 'loss', 'time_per_nth_step']
        if self.state.FLAGS.sampler == "StochasticGradientLangevinDynamics":
            header += ['scaled_gradient', 'virial', 'scaled_noise']
        elif self.state.FLAGS.sampler in ["GeometricLangevinAlgorithm_1stOrder",
                                    "GeometricLangevinAlgorithm_2ndOrder",
                                    "BAOAB",
                                    "CovarianceControlledAdaptiveLangevinThermostat"]:
            header += ['total_energy', 'kinetic_energy', 'scaled_momentum',
                      'scaled_gradient', 'virial', 'scaled_noise']
        elif "HamiltonianMonteCarlo" in self.state.FLAGS.sampler:
            header += ['total_energy', 'old_total_energy', 'kinetic_energy', 'scaled_momentum',
                      'scaled_gradient', 'virial', 'average_rejection_rate']
        return header

    def _get_initial_test_nodes(self):
        return ["sample_step"]

    def init_trajectory(self, prior, model):
        """Initialize the samplers for this trajectory.
        
        Note:
            Because of the multiple walkers that may share
            gradient and position information, we need to split
            up the typical `NeuralNetwork.add_sample_method()`
            approach.
        
            1. Prepare the samplers by adding required placeholders
            nodes and the samplers itself
            2. Prepare the gradient evaluation nodes such that
            gradients from all walkers are combined
            3. Hand the combined gradients to each sampler method
            to complete the position update nodes of the sampling
            method.

        Args:
          prior: prior for the sampler
          model: ref to the full model

        Returns:

        """
        # prepare samplers
        self.sampler = []
        for i in range(self.state.FLAGS.number_walkers):
            if self.state.FLAGS.seed is not None:
                walker_seed = self.state.FLAGS.seed + i
            else:
                walker_seed = self.state.FLAGS.seed

            # raise exception if HMC is used with multiple walkers
            if "HamiltonianMonteCarlo" in self.state.FLAGS.sampler \
                and self.state.FLAGS.number_walkers > 1:
                raise NotImplementedError(
                    "HamiltonianMonteCarlo implementation has not been properly tested with multiple walkers.")

            self.state.nn[i]._prepare_global_placeholders()
            self.sampler.append(self.state.nn[i]._prepare_sampler(
                model.loss[i], sampling_method=self.state.FLAGS.sampler,
                seed=walker_seed, prior=prior,
                sigma=self.state.FLAGS.sigma, sigmaA=self.state.FLAGS.sigmaA))

        # create combined gradients
        grads_and_vars = []
        for i in range(self.state.FLAGS.number_walkers):
            with tf.name_scope('gradients_walker'+str(i+1)):
                trainables = tf.get_collection_ref(model.trainables[i])
                grads_and_vars.append(self.sampler[i].compute_and_check_gradients(
                    model.loss[i], var_list=trainables))

        # add position update nodes
        for i in range(self.state.FLAGS.number_walkers):
            with tf.variable_scope("var_walker" + str(i + 1)):
                global_step = self.state.nn[i]._prepare_global_step()
                sample_step = self.sampler[i].apply_gradients(
                    grads_and_vars, i, global_step=global_step,
                    name=self.sampler[i].get_name())
            self.state.nn[i].summary_nodes['sample_step'] = sample_step
            self.state.nn[i].summary_nodes['EQN_step'] = self.sampler[i].EQN_update

    @staticmethod
    def get_trajectory_type():
        return "sample"

    @staticmethod
    def print_success(_):
        logging.info("SAMPLED.")

    def get_methodname(self):
        return self.state.FLAGS.sampler

    def set_beta_for_execute(self):
        self.averages._inverse_temperature = self.state.FLAGS.inverse_temperature

    def extra_evaluation_before_step(self, current_step, session, placeholder_nodes, test_nodes, feed_dict, extra_values):
        super(TrajectorySampling, self).extra_evaluation_before_step(
            current_step, session, placeholder_nodes, test_nodes, feed_dict, extra_values
        )
        # perform the EQN update step
        if self.state.FLAGS.covariance_blending != 0. and \
                                current_step % self.state.FLAGS.covariance_after_steps == 0:
            session.run(self.EQN_nodes, feed_dict=feed_dict)

    def perform_step(self, current_step, session, test_nodes, feed_dict):
        step_success = False
        blending_key = [self.state.nn[walker_index].placeholder_nodes["covariance_blending"]
                        for walker_index in range(self.state.FLAGS.number_walkers)]
        old_eta = [feed_dict[blending_key[walker_index]] for walker_index in range(self.state.FLAGS.number_walkers)]
        while not step_success:
            for walker_index in range(self.state.FLAGS.number_walkers):
                if feed_dict[blending_key[walker_index]] > 0. \
                        and feed_dict[blending_key[walker_index]] < 1e-12:
                    logging.warning("Possible NaNs or Infs in covariance matrix, setting eta to 0 temporarily.")
                    feed_dict[blending_key[walker_index]] = 0.
            try:
                summary, self.accumulated_values.accuracy, \
                self.accumulated_values.global_step, self.accumulated_values.loss = \
                    self.state._perform_step(session, test_nodes, feed_dict)
                step_success = True
            except tf.errors.InvalidArgumentError as err:
                # Cholesky failed, try again with smaller eta
                for walker_index in range(self.state.FLAGS.number_walkers):
                    feed_dict[blending_key[walker_index]] = feed_dict[blending_key[walker_index]] / 2.
                logging.warning(str(err.op) + " FAILED at step %d, using %lg as eta." \
                                % (current_step, feed_dict[blending_key[0]]))
        for walker_index in range(self.state.FLAGS.number_walkers):
            feed_dict[blending_key[walker_index]] = old_eta[walker_index]

        return summary, self.accumulated_values.accuracy, \
                self.accumulated_values.global_step, self.accumulated_values.loss
