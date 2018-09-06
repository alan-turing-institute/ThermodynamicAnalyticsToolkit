# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
import tensorflow as tf

from TATi.samplers.hamiltonianmontecarlosamplerfirstordersampler import HamiltonianMonteCarloSamplerFirstOrderSampler


class HamiltonianMonteCarloSamplerSecondOrderSampler(HamiltonianMonteCarloSamplerFirstOrderSampler):
    """ Implements a Hamiltonian Monte Carlo Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    """
    def __init__(self,
                 ensemble_precondition, step_width, inverse_temperature,
                 loss, current_step, next_eval_step, hd_steps, accept_seed,
                 seed=None, use_locking=False, name='HamiltonianMonteCarlo_2ndOrder'):
        """ Init function for this class.

        :param ensemble_precondition: whether to precondition the gradient using
                all the other walkers or not
        :param step_width: step width for gradient
        :param inverse_temperature: scale for noise
        :param loss: loss value of the current state for evaluating acceptance
        :param current_step: current step
        :param next_eval_step: step number at which accept/reject is evaluated next
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(HamiltonianMonteCarloSamplerSecondOrderSampler, self).__init__(
            ensemble_precondition, step_width, inverse_temperature,
            loss, current_step, next_eval_step, accept_seed,
            seed, use_locking, name)
        self._hd_steps = hd_steps

    def _prepare(self):
        """ Converts step width into a tensor, if given as a floating-point
        number.
        """
        super(HamiltonianMonteCarloSamplerSecondOrderSampler, self)._prepare()
        self._hd_steps_t = ops.convert_to_tensor(self._hd_steps, name="hd_steps")

    def _get_momentum_criterion_block(self, var,
                                      scaled_gradient, scaled_noise,
                                      current_step_t, next_eval_step_t, hd_steps_t):
        momentum = self.get_slot(var, "momentum")

        def momentum_id_block():
            return tf.identity(momentum)

        # update momentum
        def momentum_step_block():
            with tf.control_dependencies([
                state_ops.assign_sub(momentum, scaled_gradient)]):
                return tf.identity(momentum)

        # L=5, step 0 was a criterion evaluation:
        # 1 (BA), 2 (BBA), 3 (BBA), 4 (BBA), 5 (BBA), 6(B), 7 criterion

        # in the very first step we have to skip the first "B" step:
        # e.g., for L=5, we execute at steps 2,3,4,5,6, and skip at 1,7
        momentum_first_step_block_t = tf.cond(
            tf.logical_and(
                tf.greater_equal(current_step_t, next_eval_step_t - (hd_steps_t)),
                tf.less(current_step_t, next_eval_step_t)),
            momentum_step_block, momentum_id_block)

        # calculate kinetic energy and momentum after first "B" step
        momentum_sq = tf.reduce_sum(tf.multiply(momentum_first_step_block_t, momentum_first_step_block_t))
        momentum_global_t = HamiltonianMonteCarloSamplerFirstOrderSampler._add_momentum_contribution(momentum_sq)
        kinetic_energy_t = HamiltonianMonteCarloSamplerFirstOrderSampler._add_kinetic_energy_contribution(momentum_sq)

        def moment_reinit_block():
            with tf.control_dependencies([momentum.assign(scaled_noise)]):
                return tf.identity(momentum)

        # make sure that first momentum step (and kinetic energy) is done before second step
        def momentum_criterion_block():
            with tf.control_dependencies([momentum_global_t, kinetic_energy_t]):
                return tf.cond(
                    tf.equal(current_step_t, next_eval_step_t),
                    moment_reinit_block, momentum_step_block)

        # skip second "B" step on the extra step (as both "BA" is skipped)
        # before criterion evaluation but still make sure that kinetic
        # energy and so on is computed
        with tf.control_dependencies([momentum_global_t, kinetic_energy_t]):
            momentum_second_step_block_t = tf.cond(
                tf.equal(current_step_t, next_eval_step_t - 1),
                momentum_id_block, momentum_criterion_block)

        return momentum_second_step_block_t

    def _create_criterion_integration_block(self, var,
                                            virial_global_t,
                                            scaled_momentum, current_energy,
                                            p_accept, uniform_random_t,
                                            current_step_t, next_eval_step_t):
        initial_parameters = self.get_slot(var, "initial_parameters")
        old_total_energy_t = self._get_old_total_energy()
        accepted_t, rejected_t = self._get_accepted_rejected()

        # see https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond
        # In other words, each branch inside a tf.cond is evaluated. All "side effects"
        # need to be hidden away inside tf.control_dependencies.
        # I.E. DONT PLACE INSIDE NODES (confusing indeed)
        def accept_block():
            with tf.control_dependencies([scaled_momentum, virial_global_t]):
                with tf.control_dependencies([
                        old_total_energy_t.assign(current_energy),
                        initial_parameters.assign(var),
                        accepted_t.assign_add(1)]):
                    return tf.identity(old_total_energy_t)

        # DONT use nodes in the control_dependencies, always functions!
        def reject_block():
            with tf.control_dependencies([scaled_momentum, virial_global_t]):
                with tf.control_dependencies([
                        var.assign(initial_parameters),
                        rejected_t.assign_add(1)]):
                    return tf.identity(old_total_energy_t)

        def accept_reject_block():
            return tf.cond(tf.greater(p_accept, uniform_random_t),
                accept_block, reject_block)

        # DONT use nodes in the control_dependencies, always functions!
        def step_block():
            with tf.control_dependencies([virial_global_t]):
                with tf.control_dependencies([state_ops.assign_add(var, scaled_momentum)]):
                    return tf.identity(old_total_energy_t)

        def id_block():
            with tf.control_dependencies([virial_global_t]):
                return tf.identity(old_total_energy_t)

        # skip "A" step in extra step before criterion evaluation
        def step_or_id_block():
           return tf.cond(
                tf.equal(current_step_t, next_eval_step_t - 1),
               id_block, step_block)

        # make sure virial and gradients are evaluated before we update variables
        criterion_block_t = tf.cond(
            tf.equal(current_step_t, next_eval_step_t),
            accept_reject_block, step_or_id_block)

        return criterion_block_t

    def _apply_dense(self, grads_and_vars, var):
        """ Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling.

        We perform a number of Leapfrog steps on a hamiltonian (loss+kinetic energy)
        and at step number next_eval_step we check the acceptance criterion,
        either resetting back to the initial parameters or resetting the
        initial parameters to the current ones.

        NOTE:
            Due to Tensorflow enforcing loss and gradient evaluation at
            the begin of the sampling step, we need to cyclically permute the
            BAB steps to become BBA, i.e. the last "B" step is delayed till the
            next step. This means that we need to skip the additional "B" step
            in the very first time integration step and we need an additional
            step to compute the delayed "B" for the last time integration and
            subsequently to compute the kinetic energy before the criterion
            evaluation.

            Effectively, we compute L+2 steps if L is the number of Hamiltonian
            dynamics steps.

        :param grads_and_vars: gradient nodes over all walkers and all variables
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        grad = self._pick_grad(grads_and_vars, var)
        step_width_t, inverse_temperature_t, current_step_t, next_eval_step_t, random_noise_t, uniform_random_t = \
            self._prepare_dense(grad, var)
        hd_steps_t =  math_ops.cast(self._hd_steps_t, tf.int64)

        # 1/2 * \nabla V (q^n ) \Delta t
        scaled_gradient = .5 * step_width_t * grad

        gradient_global_t = self._add_gradient_contribution(scaled_gradient)
        virial_global_t = self._add_virial_contribution(grad, var)

        # update momentum: B, BB or redraw momenta
        scaled_noise = tf.sqrt(1./inverse_temperature_t)*random_noise_t
        momentum_criterion_block_t = self._get_momentum_criterion_block(var,
            scaled_gradient, scaled_noise, current_step_t, next_eval_step_t, hd_steps_t)

        current_energy = self._get_current_total_energy()

        # prior force act directly on var
        #ub_repell, lb_repell = self._apply_prior(var)
        #prior_force = step_width_t * (ub_repell + lb_repell)

        #scaled_momentum = step_width_t * momentum_criterion_block_t - prior_force

        # update variables: A, skip or evaluate criterion (accept/reject)
        scaled_momentum = step_width_t * momentum_criterion_block_t
        p_accept = self._create_p_accept(inverse_temperature_t, current_energy)
        criterion_block_t = self._create_criterion_integration_block(var,
            virial_global_t, scaled_momentum, current_energy,
            p_accept, uniform_random_t,
            current_step_t, next_eval_step_t
        )

        # note: these are evaluated in any order, use control_dependencies if required
        return control_flow_ops.group(*([momentum_criterion_block_t, criterion_block_t,
                                        virial_global_t, gradient_global_t]))

    def _apply_sparse(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of sparsely
        occupied tensors to perform the actual sampling.

        Note that this is not implemented so far.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        raise NotImplementedError("Sparse gradient updates are not supported.")
