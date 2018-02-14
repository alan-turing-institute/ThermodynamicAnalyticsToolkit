# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
import tensorflow as tf

from DataDrivenSampler.models.basetype import dds_basetype
from DataDrivenSampler.samplers.sgldsampler import SGLDSampler


class HamiltonianMonteCarloSampler(SGLDSampler):
    """ Implements a Hamiltonian Monte Carlo Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    """
    def __init__(self, step_width, inverse_temperature, current_step, next_eval_step, accept_seed, seed=None, use_locking=False, name='HamiltonianMonteCarlo'):
        """ Init function for this class.

        :param step_width: step width for gradient
        :param inverse_temperature: scale for noise
        :param current_step: current step
        :param next_eval_step: step number at which accept/reject is evaluated next
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(HamiltonianMonteCarloSampler, self).__init__(step_width, inverse_temperature,
                                                           seed, use_locking, name)
        self._accept_seed = accept_seed
        self._current_step = current_step
        self._next_eval_step = next_eval_step

    def _prepare(self):
        """ Converts step width into a tensor, if given as a floating-point
        number.
        """
        super(HamiltonianMonteCarloSampler, self)._prepare()
        self._current_step_t = ops.convert_to_tensor(self._current_step, name="current_step")
        self._next_eval_step_t = ops.convert_to_tensor(self._next_eval_step, name="next_eval_step")

    def _create_slots(self, var_list):
        """ Slots are internal resources for the Optimizer to store values
        that are required and modified during each iteration.

        Here, we need a slot to store the parameters for the starting step of
        the short reject/accept trajectory run.

        :param var_list: list of variables
        """
        for v in var_list:
            self._zeros_slot(v, "initial_parameters", self._name)
            # we reinitialize momenta in first step anyway
            self._zeros_slot(v, "momentum", self._name)
            # # initialize with normal distribution scaled by temperature
            # #self._seed += 1
            # mom_initializer = init_ops.random_normal_initializer(
            #     dtype=v.dtype.base_dtype,
            #     mean=0., stddev=tf.sqrt(self.temperature),
            #     seed=self._seed)
            # self._get_or_make_slot_with_initializer(v, mom_initializer, v.shape,
            #                                         dtype=v.dtype.base_dtype,
            #                                         slot_name="momentum",
            #                                         op_name=self._name)

    def _prepare_dense(self, grad, var):
        """ Stuff common to all Langevin samplers.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: step_width, inverse_temperature, and noise tensors
        """
        step_width_t, inverse_temperature_t, random_noise_t = \
            super(HamiltonianMonteCarloSampler, self)._prepare_dense(grad, var)
        current_step_t = math_ops.cast(self._current_step_t, tf.int64)
        next_eval_step_t = math_ops.cast(self._next_eval_step_t, tf.int64)

        uniform_random_t = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=dds_basetype, seed=self._accept_seed)
        return step_width_t, inverse_temperature_t, current_step_t, next_eval_step_t, random_noise_t, uniform_random_t

    def _apply_dense(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling.

        We perform a leap-frog step on a hamiltonian (loss+kinetic energy)
        and at step number next_eval_step we check the acceptance criterion,
        either resetting back to the initial parameters or resetting the
        initial parameters to the current ones.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        step_width_t, inverse_temperature_t, current_step_t, next_eval_step_t, random_noise_t, uniform_random_t = \
            self._prepare_dense(grad, var)
        momentum = self.get_slot(var, "momentum")
        initial_parameters = self.get_slot(var, "initial_parameters")

        # \nabla V (q^n ) \Delta t
        scaled_gradient = step_width_t * grad

        with tf.variable_scope("accumulate", reuse=True):
            old_total_energy_t = tf.get_variable("total_energy", dtype=dds_basetype)

        with tf.variable_scope("accumulate", reuse=True):
            gradient_global = tf.get_variable("gradients", dtype=dds_basetype)
            gradient_global_t = tf.assign_add(
                gradient_global,
                tf.reduce_sum(tf.multiply(scaled_gradient, scaled_gradient)))
            # configurational temperature
            virial_global = tf.get_variable("virials", dtype=dds_basetype)
            virial_global_t = tf.assign_add(
                virial_global,
                tf.reduce_sum(tf.multiply(grad, var)))

        # update momentum
        scaled_noise = tf.sqrt(1./inverse_temperature_t)*random_noise_t
        def momentum_step_block():
            with tf.control_dependencies([state_ops.assign_sub(momentum, scaled_gradient)]):
                return tf.identity(momentum)

        def moment_reinit_block():
            with tf.control_dependencies([momentum.assign(scaled_noise)]):
                return tf.identity(momentum)

        momentum_criterion_block_t = tf.cond(
            tf.equal(current_step_t, next_eval_step_t),
            moment_reinit_block, momentum_step_block)

        with tf.variable_scope("accumulate", reuse=True):
            momentum_global = tf.get_variable("momenta", dtype=dds_basetype)
            momentum_global_t = tf.assign_add(
                momentum_global,
                tf.reduce_sum(tf.multiply(momentum_criterion_block_t, momentum_criterion_block_t)))

        momentum_sq = 0.5 * tf.reduce_sum(tf.multiply(momentum_criterion_block_t, momentum_criterion_block_t))
        with tf.variable_scope("accumulate", reuse=True):
            kinetic_energy = tf.get_variable("kinetic", dtype=dds_basetype)
            kinetic_energy_t = tf.assign_add(kinetic_energy, momentum_sq)

        with tf.variable_scope("accumulate", reuse=True):
            loss = tf.get_variable("old_loss", dtype=dds_basetype)
            kinetic_energy = tf.get_variable("old_kinetic", dtype=dds_basetype)
            current_energy = loss + kinetic_energy

        with tf.variable_scope("accumulate", reuse=True):
            accepted_t = tf.get_variable("accepted", dtype=tf.int64)
            rejected_t = tf.get_variable("rejected", dtype=tf.int64)

        # Note that it does not matter which layer actually sets the old_total_energy
        # on acceptance. As soon as the first set of variables has done it, old and
        # current are the same, hence exp(0)=1, and we always accept throughout the
        # step
        # Moreover, as all have the same seed, i.e. all get the same random number
        # sequences. Each one set of variable would accept if it were the first to
        # get called.

        # see https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond
        # In other words, each branch inside a tf.cond is evaluated. All "side effects"
        # need to be hidden away inside tf.control_dependencies.
        # I.E. DONT PLACE INSIDE NODES (confusing indeed)
        def accept_block():
            with tf.control_dependencies([old_total_energy_t.assign(current_energy),
                                          initial_parameters.assign(var),
                                          accepted_t.assign_add(1)]):
                return tf.identity(old_total_energy_t)

        # DONT use nodes in the control_dependencies, always functions!
        def reject_block():
            with tf.control_dependencies([var.assign(initial_parameters),
                                          rejected_t.assign_add(1)]):
                return tf.identity(old_total_energy_t)

        max_value_t = tf.constant(1.0, dtype=dds_basetype)
        p_accept = tf.minimum(max_value_t, tf.exp(old_total_energy_t - current_energy))

        def accept_reject_block():
            return tf.cond(
                tf.greater(p_accept, uniform_random_t),
                accept_block, reject_block)

        # prior force act directly on var
        ub_repell, lb_repell = self._apply_prior(var)
        prior_force = step_width_t * (ub_repell + lb_repell)

        # update variables
        scaled_momentum = step_width_t * momentum_criterion_block_t - prior_force

        # DONT use nodes in the control_dependencies, always functions!
        def step_block():
            with tf.control_dependencies([state_ops.assign_add(var, scaled_momentum)]):
                return tf.identity(old_total_energy_t)

        # make sure virial and gradients are evaluated before we update variables
        with tf.control_dependencies([virial_global_t, gradient_global_t]):
            criterion_block_t = tf.cond(
                tf.equal(current_step_t, next_eval_step_t),
                accept_reject_block, step_block)

        # note: these are evaluated in any order, use control_dependencies if required
        return control_flow_ops.group(*([momentum_criterion_block_t, criterion_block_t,
                                        virial_global_t, gradient_global_t,
                                        momentum_global_t, kinetic_energy_t]))

    def _apply_sparse(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of sparsely
        occupied tensors to perform the actual sampling.

        Note that this is not implemented so far.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        raise NotImplementedError("Sparse gradient updates are not supported.")
