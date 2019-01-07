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
import numpy as np
import tensorflow as tf
import time

try:
    from tqdm import tqdm # allows progress bar
    tqdm_present = True
    # workaround: otherwise we get deadlock on exceptions,
    # see https://github.com/tqdm/tqdm/issues/469
    tqdm.monitor_interval = 0
except ImportError:
    tqdm_present = False

from TATi.models.trajectories.trajectory_sampling import TrajectorySampling


class TrajectorySamplingHamiltonian(TrajectorySampling):
    """ This implements sampling of a trajectory using Hamiltonian dynamics.

    """
    def __init__(self, trajectory_state):
        super(TrajectorySamplingHamiltonian, self).__init__(trajectory_state)

    def execute(self, session, dataset_dict, return_run_info=False, return_trajectories=False, return_averages=False):
        """ Performs the actual sampling of the neural network `nn` given a dataset `ds` and a
        Session `session`.

        :param dataset_dict: contains input_pipeline, placeholders for x and y
        :param return_run_info: if set to true, run information will be accumulated
                inside a numpy array and returned
        :param return_trajectories: if set to true, trajectories will be accumulated
                inside a numpy array and returned (adhering to FLAGS.every_nth)
        :param return_averages: if set to true, accumulated average values will be
                returned as numpy array
        :return: either thrice None or lists (per walker) of pandas dataframes
                depending on whether either parameter has evaluated to True
        """

        self.init_files()

        HMC_placeholder_nodes = [self.state.nn[walker_index].get_dict_of_nodes(
            ["current_step", "next_eval_step", "step_width", "hamiltonian_dynamics_steps"])
            for walker_index in range(self.state.FLAGS.number_walkers)]

        test_nodes = self._get_test_nodes()
        EQN_nodes =[self.state.nn[walker_index].get("EQN_step") \
                           for walker_index in range(self.state.FLAGS.number_walkers)]
        all_weights, all_biases = self.state._get_all_parameters()

        self.averages.reset(return_averages=return_averages,
                            header=self.get_averages_header())
        self.averages.init_writer(self.averages_writer)
        self.averages._inverse_temperature = self.state.FLAGS.inverse_temperature
        self.run_info.reset(return_run_info=return_run_info,
                            header=self.get_run_header())
        self.run_info.init_writer(self.run_writer)
        self.trajectory.reset(return_trajectories=return_trajectories,
                              header=self._get_trajectory_header())
        self.trajectory.init_writer(self.trajectory_writer)
        self.accumulated_values.reset()

        # place in feed dict: We have to supply all placeholders (regardless of
        # which the employed sampler actually requires) because of the evaluated
        # summary! All of the placeholder nodes are also summary nodes.
        feed_dict = {}
        for walker_index in range(self.state.FLAGS.number_walkers):
            feed_dict.update(self.state._create_default_feed_dict_with_constants(walker_index))

        # zero extra nodes for HMC
        self._prepare_HMC_nodes(session)

        # check that sampler's parameters are actually used
        self._print_parameters(session, feed_dict)

        # prepare summaries for TensorBoard
        summary_writer = self.state._prepare_summaries(session)

        # prepare some loop variables
        logging.info("Starting to sample")
        logging.info_intervals = max(1, int(self.state.FLAGS.max_steps / 100))
        self.state.last_time = time.time()
        self.state.elapsed_time = 0.
        HD_steps = [-2]*self.state.FLAGS.number_walkers        # number of hamiltonian dynamics steps
        HMC_steps = [0]*self.state.FLAGS.number_walkers       # next step where to evaluate criterion
        HMC_old_steps = [0]*self.state.FLAGS.number_walkers   # last step where criterion was evaluated
        # we need to randomly vary the step widths to avoid (quasi-)periodicities
        step_widths = [self.state.FLAGS.step_width] * self.state.FLAGS.number_walkers
        self._set_HMC_placeholders(HMC_placeholder_nodes,
                                   1, step_widths, HD_steps, HMC_steps, feed_dict)
        if tqdm_present and self.state.FLAGS.progress:
            step_range = tqdm(range(self.state.FLAGS.max_steps))
        else:
            step_range = range(self.state.FLAGS.max_steps)

        # backup gradients and virials of each initial state to avoid recalculation
        initial_state_gradients = [None]*self.state.FLAGS.number_walkers
        initial_state_virials = [None]*self.state.FLAGS.number_walkers
        initial_state_inertia = [None]*self.state.FLAGS.number_walkers
        initial_state_momenta = [None]*self.state.FLAGS.number_walkers

        last_rejected = session.run(self.state.static_vars["rejected"])   # temporary to see whether last evaluation was a rejection
        for current_step in step_range:
            # get next batch of data
            features, labels = dataset_dict["input_pipeline"].next_batch(session)
            # logging.debug("batch is x: "+str(features[:])+", y: "+str(labels[:]))

            # update feed_dict for this step
            feed_dict.update({
                dataset_dict["xinput"]: features,
                dataset_dict["true_labels"]: labels
            })

            # get energies for acceptance evaluation in very first step
            if current_step == 0:
                self.accumulated_values.loss, self.accumulated_values.kinetic_energy = \
                    session.run([test_nodes[3], self.state.static_vars["kinetic_energy"]], feed_dict=feed_dict)

            # set global variable used in HMC sampler for criterion to initial loss
            self._set_HMC_eval_variables(session, current_step, HMC_steps, self.accumulated_values)

            # zero kinetic energy and other variables
            self.state._zero_state_variables(session, self.state.FLAGS.sampler)

            # tell accumulators about next evaluation step (delayed by one)
            self.run_info.inform_next_eval_step(HMC_steps, self.accumulated_values.rejected)
            self.trajectory.inform_next_eval_step(HMC_steps, self.accumulated_values.rejected)
            self.averages.inform_next_eval_step(HMC_steps, self.accumulated_values.rejected)

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
            if current_step % self.state.FLAGS.every_nth == 0:
                self.accumulated_values.weights, self.accumulated_values.biases = \
                    self._get_parameters(session, return_trajectories, all_weights, all_biases)

            # perform the EQN update step
            if self.state.FLAGS.covariance_blending != 0. and \
                    current_step % self.state.FLAGS.covariance_after_steps == 0:
                session.run(EQN_nodes, feed_dict=feed_dict)

            # perform the sampling step
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
                        feed_dict[blending_key[walker_index]] = feed_dict[blending_key[walker_index]]/2.
                    logging.warning(str(err.op) + " FAILED at step %d, using %lg as eta." \
                                    % (current_step, feed_dict[blending_key[0]]))
            for walker_index in range(self.state.FLAGS.number_walkers):
                feed_dict[blending_key[walker_index]] = old_eta[walker_index]

            # get updated state variables
                self.accumulated_values.evaluate(session, self.state.FLAGS.sampler, self.state.static_vars)

            def print_energies():
                if self.state.FLAGS.verbose > 1:
                    for walker_index in range(self.state.FLAGS.number_walkers):
                        loss_subtext = "n" if self.accumulated_values.rejected[walker_index] != last_rejected[walker_index] else "n-1"
                        kinetic_subtext = "n-1" if "HamiltonianMonteCarlo_2ndOrder" in self.state.FLAGS.sampler \
                                and current_step != HMC_old_steps[walker_index] else "n"
                        logging.debug("walker #%d, #%d: L(x_{%s})=%lg, total is %lg, T(p_{%s})=%lg, sum is %lg" \
                                  % (walker_index, current_step, loss_subtext,
                                     # to emphasize updated loss
                                     self.accumulated_values.loss[walker_index],
                                     self.accumulated_values.old_total_energy[walker_index],
                                     kinetic_subtext, # for HMC_2nd
                                     self.accumulated_values.kinetic_energy[walker_index],
                                     self.accumulated_values.loss[walker_index] + self.accumulated_values.kinetic_energy[
                                         walker_index]))

            # give output on debug mode
            print_energies()

            # if last step was rejection, re-evaluate loss and weights as state changed
            if self.accumulated_values.rejected != last_rejected:
                # recalculate loss and get kinetic energy
                self.accumulated_values.loss, self.accumulated_values.kinetic_energy = \
                    session.run([test_nodes[3], self.state.static_vars["kinetic_energy"]], feed_dict=feed_dict)
                # get restored weights and biases
                self.accumulated_values.weights, self.accumulated_values.biases = \
                    self._get_parameters(session, return_trajectories, all_weights, all_biases)
                logging.info("Last state REJECTed.")
                print_energies()

            # reset gradients and virials to initial state's if rejected
            for walker_index in range(self.state.FLAGS.number_walkers):
                if current_step == HMC_old_steps[walker_index]:
                    # restore gradients and virials
                    if self.accumulated_values.rejected[walker_index] != last_rejected[walker_index]:
                        self.accumulated_values.gradients[walker_index] = initial_state_gradients[walker_index]
                        self.accumulated_values.virials[walker_index] = initial_state_virials[walker_index]
                        self.accumulated_values.inertia[walker_index] = initial_state_inertia[walker_index]
                        self.accumulated_values.momenta[walker_index] = initial_state_momenta[walker_index]
                    else:
                        initial_state_gradients[walker_index] = self.accumulated_values.gradients[walker_index]
                        initial_state_virials[walker_index] = self.accumulated_values.virials[walker_index]
                        initial_state_inertia[walker_index] = self.accumulated_values.inertia[walker_index]
                        initial_state_momenta[walker_index] = self.accumulated_values.momenta[walker_index]
                    # accumulate averages and other information
                    if current_step >= self.state.FLAGS.burn_in_steps:
                        self.averages.accumulate_each_step(current_step, walker_index, self.accumulated_values)

            # write summaries for tensorboard
            self.state._write_summaries(summary_writer, summary, current_step)

            if current_step % self.state.FLAGS.every_nth == 0:
                self.accumulated_values.time_elapsed_per_nth_step = self.state._get_elapsed_time_per_nth_step(current_step)

            for walker_index in range(self.state.FLAGS.number_walkers):
                self.run_info.accumulate_nth_step(current_step, walker_index,
                                                  self.accumulated_values)
                self.trajectory.accumulate_nth_step(current_step, walker_index,
                                                    self.accumulated_values)
                self.averages.accumulate_nth_step(current_step, walker_index,
                                                  self.accumulated_values)

            # update temporary rejected
            last_rejected = self.accumulated_values.rejected

            #if (i % logging.info_intervals) == 0:
                #logging.debug('Accuracy at step %s (%s): %s' % (i, global_step, acc))
                #logging.debug('Loss at step %s: %s' % (i, loss_eval))

            self.state._decide_collapse_walkers(session, current_step)

        logging.info("SAMPLED.")

        # close summaries file
        if self.state.FLAGS.summaries_path is not None:
            summary_writer.close()

        self.close_files()

        return self.run_info.run_info, self.trajectory.trajectory, self.averages.averages

    def _set_HMC_placeholders(self, HMC_placeholder_nodes, current_step, step_widths, HD_steps, HMC_steps, feed_dict):
        if "HamiltonianMonteCarlo" in self.state.FLAGS.sampler:
            for walker_index in range(self.state.FLAGS.number_walkers):
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
        if "HamiltonianMonteCarlo" in self.state.FLAGS.sampler:
            for walker_index in range(self.state.FLAGS.number_walkers):
                if current_step > HMC_steps[walker_index]:
                    # pick next evaluation step with a little random variation
                    step_widths[walker_index] = \
                        np.random.uniform(low=0.7, high=1.3) * self.state.FLAGS.step_width
                    logging.debug("Next step width of #"+str(walker_index) \
                                  +" is " + str(step_widths[walker_index]))

                    # pick next evaluation step with a little random variation
                    HD_steps[walker_index] = \
                        max(1, round((0.9 + np.random.uniform(low=0., high=0.2)) \
                                     * self.state.FLAGS.hamiltonian_dynamics_time / self.state.FLAGS.step_width))
                    if self.state.FLAGS.sampler == "HamiltonianMonteCarlo_1stOrder":
                        # one extra step for the criterion evaluation
                        HMC_steps[walker_index] += 1 + HD_steps[walker_index]
                    elif self.state.FLAGS.sampler == "HamiltonianMonteCarlo_2ndOrder":
                        # with Leapfrog integration we need an additional step
                        # for the last "B" step of BAB due to cyclic permutation
                        # to BBA.
                        HMC_steps[walker_index] += 2 + HD_steps[walker_index]
                    else:
                        raise NotImplementedError("The HMC sampler method %S is unknown" % (self.state.FLAGS.sampler))
                    logging.debug("Next amount of HD steps is " + str(HD_steps)
                                  +", evaluation of HMC criterion at step " + str(HMC_steps))
        else:
            for walker_index in range(self.state.FLAGS.number_walkers):
                HMC_steps[walker_index] = current_step
        return HD_steps, HMC_steps

    def _set_HMC_eval_variables(self, session, current_step, HMC_steps, values):
        if "HamiltonianMonteCarlo" in self.state.FLAGS.sampler:
            # set current kinetic as it is accumulated outside of tensorflow
            kin_eval = session.run(self.state.static_vars["kinetic_energy"])
            set_dict = {}
            for walker_index in range(self.state.FLAGS.number_walkers):
                set_dict[ self.state.placeholders["current_kinetic"][walker_index]] = \
                    kin_eval[walker_index]
            session.run(self.state.assigner["current_kinetic"], feed_dict=set_dict)

            # possibly reset some old energy values for acceptance criterion if
            # an acceptance evaluation has just occured
            #
            # Note that we always evaluate the criterion in step 0 and make sure that
            # the we always accept (by having old total and current energy coincide)
            # such that a valid old parameter set is stored to which we may restore
            # if the next evaluation rejects.
            do_evaluate = False
            for walker_index in range(self.state.FLAGS.number_walkers):
                if current_step > HMC_steps[walker_index] or current_step == 0:
                    # at least one walker requires a loss calculation
                    do_evaluate = True
            if do_evaluate or self.state.FLAGS.verbose > 1:
                HMC_set_total_energy = []
                energy_placeholders = {}
                for walker_index in range(self.state.FLAGS.number_walkers):
                    if current_step > HMC_steps[walker_index] or current_step == 0:
                        HMC_set_total_energy.extend([self.state.assigner["old_total_energy"][walker_index]])
                        energy_placeholders[self.state.placeholders["old_total_energy"][walker_index]] = \
                            values.loss[walker_index]+values.kinetic_energy[walker_index]
                        logging.debug("Resetting total energy for walker #"+str(walker_index))
                if len(HMC_set_total_energy) > 0:
                    total_eval = session.run(HMC_set_total_energy, feed_dict=energy_placeholders)
                    logging.debug("New total energies are "+str(total_eval))

    def _prepare_HMC_nodes(self, session):
        if "HamiltonianMonteCarlo" in self.state.FLAGS.sampler:
            # zero rejection rate before sampling start
            check_accepted, check_rejected = session.run([
                self.state.zero_assigner["accepted"], self.state.zero_assigner["rejected"]])
            for walker_index in range(self.state.FLAGS.number_walkers):
                assert(check_accepted[walker_index] == 0)
                assert(check_rejected[walker_index] == 0)


