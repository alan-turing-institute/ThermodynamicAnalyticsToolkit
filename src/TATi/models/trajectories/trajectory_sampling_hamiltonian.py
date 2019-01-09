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

    Due to the Metropolis-Hastings criterion it behaves quite differently
    compared to a Langevin dynamics based sampler. Therefore a number
    of extra functions are needed for the book-keeping of all values
    associated with the criterion evaluation.

    """
    def __init__(self, trajectory_state):
        super(TrajectorySamplingHamiltonian, self).__init__(trajectory_state)

    def _set_HMC_placeholders(self, HMC_placeholder_nodes, current_step, step_widths, HD_steps, HMC_steps, feed_dict):
        """ Updates feed_dict with extra values for HMC

        :param HMC_placeholder_nodes: keys to extra values
        :param current_step: current step
        :param step_widths: step widths per walker
        :param HD_steps: current number of Hamiltonian Dynamics steps per walker
        :param HMC_steps: current step numbers (one per walker) when to evaluate criterion
        :param feed_dict: feed dict for `tf.Session.run()`
        """
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

    def _set_HMC_next_eval_step(self, current_step, step_widths, HD_steps, HMC_steps):
        """ Compute when next to evaluate HMC acceptance criterion

        In order to avoid correlations between different walkers, it is
        recommended [Neal, 2011] to randomly vary the step widths and
        the exact step when to evaluate the criterion. This is done
        here for `step_widths`, `HD_steps`, and `HMC_steps`.


        :param current_step: current step
        :param step_widths: step widths per walker
        :param HD_steps: current number of Hamiltonian Dynamics steps per walker
        :param HMC_steps: current step numbers (one per walker) when to evaluate criterion
        """
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

    def _set_HMC_eval_variables(self, session, current_step, HMC_steps, values):
        """ Set values in feed_dict needed for HMC's criterion evaluation.

        Some values are accumulated outside tensorflow's computational graph
        and need to be piped back into the graph through placeholder nodes.

        However, we need to update them only right after the evaluation
        criterion has been computed.

        :param session: `tf.Session` object
        :param current_step:
        :param HMC_steps: step numbers (one per walker) when to evaluate criterion
        :param values: accumulated values with updated energies
        """
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

    def zero_extra_nodes(self, session):
        """ Zero accepted and rejected counters before the actual iteration.

        :param session: `tf.Session` object
        """
        if "HamiltonianMonteCarlo" in self.state.FLAGS.sampler:
            # zero rejection rate before sampling start
            check_accepted, check_rejected = session.run([
                self.state.zero_assigner["accepted"], self.state.zero_assigner["rejected"]])
            for walker_index in range(self.state.FLAGS.number_walkers):
                assert(check_accepted[walker_index] == 0)
                assert(check_rejected[walker_index] == 0)

    def get_placeholder_nodes(self):
        """ Get HMC specific nodes needed as keys in feed_dict.

        :return: list of extra nodes
        """
        retlist = super(TrajectorySamplingHamiltonian, self).get_placeholder_nodes()
        retlist.extend( [self.state.nn[walker_index].get_dict_of_nodes(
            ["current_step", "next_eval_step", "step_width", "hamiltonian_dynamics_steps"])
            for walker_index in range(self.state.FLAGS.number_walkers)])
        return retlist

    def prepare_extra_values(self, placeholder_nodes, session, feed_dict):
        """ Prepare a set of temporary values in a dict.

        :param placeholder_nodes: extra HMC nodes, used as keys to update in `feed_dict`
        :param session: `tf.Session` object
        :param feed_dict: feed dict for `tf.Session.run()`
        :return: dict with the extra values
        """
        array = {}
        array["HD_steps"] = [-2]*self.state.FLAGS.number_walkers        # number of hamiltonian dynamics steps
        array["HMC_steps"] = [0]*self.state.FLAGS.number_walkers       # next step where to evaluate criterion
        array["HMC_old_steps"] = [0]*self.state.FLAGS.number_walkers   # last step where criterion was evaluated
        # we need to randomly vary the step widths to avoid (quasi-)periodicities
        array["step_widths"] = [self.state.FLAGS.step_width] * self.state.FLAGS.number_walkers
        self._set_HMC_placeholders(placeholder_nodes,
                                   1, array["step_widths"], array["HD_steps"],
                                   array["HMC_steps"], feed_dict)

        # backup gradients and virials of each initial state to avoid recalculation
        array["initial_state_gradients"] = [None]*self.state.FLAGS.number_walkers
        array["initial_state_virials"] = [None]*self.state.FLAGS.number_walkers
        array["initial_state_inertia"] = [None]*self.state.FLAGS.number_walkers
        array["initial_state_momenta"] = [None]*self.state.FLAGS.number_walkers

        # remember rejected whose change indicates when to reset state
        array["last_rejected"] = session.run(self.state.static_vars["rejected"])

        return array

    def extra_evaluation_before_step(self, current_step, session, placeholder_nodes, test_nodes, feed_dict, extra_values):
        """ Before th actual update step, update HMC's specific variables

        :param current_step: current step
        :param session: `tf.Session` object
        :param placeholder_nodes: extra HMC nodes, used as keys to update in `feed_dict`
        :param test_nodes: nodes to evaluate for loss and others
        :param feed_dict: feed dict for `tf.Session.run()`
        :param extra_values: temporary values dict used during iteration
        """
        # get energies for acceptance evaluation in very first step
        if current_step == 0:
            self.accumulated_values.loss, self.accumulated_values.kinetic_energy = \
                session.run([test_nodes[3], self.state.static_vars["kinetic_energy"]], feed_dict=feed_dict)

        # set global variable used in HMC sampler for criterion to initial loss
        self._set_HMC_eval_variables(session, current_step, extra_values["HMC_steps"], self.accumulated_values)

        # tell accumulators about next evaluation step (delayed by one)
        self.run_info.inform_next_eval_step(extra_values["HMC_steps"], self.accumulated_values.rejected)
        self.trajectory.inform_next_eval_step(extra_values["HMC_steps"], self.accumulated_values.rejected)
        self.averages.inform_next_eval_step(extra_values["HMC_steps"], self.accumulated_values.rejected)

        # set next criterion evaluation step
        # needs to be after `_set_HMC_eval_variables()`
        # needs to be before `perform_step()`
        extra_values["HMC_old_steps"][:] = extra_values["HMC_steps"]
        self._set_HMC_next_eval_step(
            current_step, extra_values["step_widths"], extra_values["HD_steps"], extra_values["HMC_steps"])
        self._set_HMC_placeholders(
            placeholder_nodes,
            current_step,
            extra_values["step_widths"],extra_values["HD_steps"], extra_values["HMC_steps"],
            feed_dict)

    def print_energies(self, current_step, extra_values):
        """ Print HMC energies to make update step understandable

        :param current_step: current step
        :param extra_values: temporary values dict used during iteration
        """
        if self.state.FLAGS.verbose > 1:
            for walker_index in range(self.state.FLAGS.number_walkers):
                loss_subtext = "n" if self.accumulated_values.rejected[walker_index] != extra_values["last_rejected"][
                    walker_index] else "n-1"
                kinetic_subtext = "n-1" if "HamiltonianMonteCarlo_2ndOrder" in self.state.FLAGS.sampler \
                                           and current_step != extra_values["HMC_old_steps"][walker_index] else "n"
                logging.debug("walker #%d, #%d: L(x_{%s})=%lg, total is %lg, T(p_{%s})=%lg, sum is %lg" \
                              % (walker_index, current_step, loss_subtext,
                                 # to emphasize updated loss
                                 self.accumulated_values.loss[walker_index],
                                 self.accumulated_values.old_total_energy[walker_index],
                                 kinetic_subtext,  # for HMC_2nd
                                 self.accumulated_values.kinetic_energy[walker_index],
                                 self.accumulated_values.loss[walker_index] + self.accumulated_values.kinetic_energy[
                                     walker_index]))

    def update_values(self, current_step, session, test_nodes, feed_dict, extra_values, parameters_list):
        """ Extra update for HMC in order to reset state if necessary.

        If this step has been a criterion evaluation, then we need to account for
        the resetted state if the proposed one was rejected. Moreover, we need
        to re-calculate the energies.

        :param current_step: current step
        :param session: `tf.Session` object
        :param test_nodes: nodes to evaluate for loss and others
        :param feed_dict: feed dict for `tf.Session.run()`
        :param extra_values: temporary values dict used during iteration
        :param parameters_list: parameters needed to obtained updated weights and
                biases.
        """
        # if last step was rejection, re-evaluate loss and weights as state changed
        if self.accumulated_values.rejected != extra_values["last_rejected"]:
            # recalculate loss and get kinetic energy
            self.accumulated_values.loss, self.accumulated_values.kinetic_energy = \
                session.run([test_nodes[3], self.state.static_vars["kinetic_energy"]], feed_dict=feed_dict)
            # get restored weights and biases
            self.accumulated_values.weights, self.accumulated_values.biases = \
                self._get_parameters(session, *parameters_list)
            logging.info("Last state REJECTed.")
            self.print_energies(current_step, extra_values)

        # reset gradients and virials to initial state's if rejected
        for walker_index in range(self.state.FLAGS.number_walkers):
            if current_step == extra_values["HMC_old_steps"][walker_index]:
                # restore gradients and virials
                if self.accumulated_values.rejected[walker_index] != extra_values["last_rejected"][walker_index]:
                    self.accumulated_values.gradients[walker_index] = extra_values["initial_state_gradients"][walker_index]
                    self.accumulated_values.virials[walker_index] = extra_values["initial_state_virials"][walker_index]
                    self.accumulated_values.inertia[walker_index] = extra_values["initial_state_inertia"][walker_index]
                    self.accumulated_values.momenta[walker_index] = extra_values["initial_state_momenta"][walker_index]
                else:
                    extra_values["initial_state_gradients"][walker_index] = self.accumulated_values.gradients[walker_index]
                    extra_values["initial_state_virials"][walker_index] = self.accumulated_values.virials[walker_index]
                    extra_values["initial_state_inertia"][walker_index] = self.accumulated_values.inertia[walker_index]
                    extra_values["initial_state_momenta"][walker_index] = self.accumulated_values.momenta[walker_index]
                # accumulate averages and other information
                if current_step >= self.state.FLAGS.burn_in_steps:
                    self.averages.accumulate_each_step(current_step, walker_index, self.accumulated_values)

    def update_averages(self, current_step):
        """ Do not update averages in each step for HMC.

        With HMC we are only interested in the final accepted or rejected
        new states. We do not care about the intermediate steps obtained
        through Hamiltonian dynamics. Therefore, we cannot simply call
        this for every step but only for the criterion evaluation steps.,
        see `TrajectorySamplingHamiltonian.update_values()`.

        :param current_step: current step
        """
        pass

    def update_extra_values(self, extra_values):
        """ Store the possibly updated rejected value in our temporary
        in `extra_values`.

        :param extra_values: extra values containing temporaries used
            during the iteration
        """
        extra_values["last_rejected"] = self.accumulated_values.rejected
