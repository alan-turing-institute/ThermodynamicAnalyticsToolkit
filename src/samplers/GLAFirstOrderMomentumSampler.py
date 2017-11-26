# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
import tensorflow as tf

from DataDrivenSampler.samplers.sgldsampler import SGLDSampler


class GLAFirstOrderMomentumSampler(SGLDSampler):
    """ Implements a Geometric Langevin Algorithm 1st order
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    """
    def __init__(self, step_width, inverse_temperature, friction_constant,
                 seed=None, use_locking=False, name='GLA_1stOrder'):
        """ Init function for this class.

        :param step_width: step width for gradient, also affects inject noise
        :param inverse_temperature: scale for gradients
        :param friction_constant: scales the momenta
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(GLAFirstOrderMomentumSampler, self).__init__(step_width, inverse_temperature,
                                                           seed, use_locking, name)
        self._friction_constant = friction_constant

    def _prepare(self):
        """ Converts step width into a tensor, if given as a floating-point
        number.
        """
        super(GLAFirstOrderMomentumSampler, self)._prepare()
        self._friction_constant_t = ops.convert_to_tensor(self._friction_constant, name="friction_constant")

    def _create_slots(self, var_list):
        """ Slots are internal resources for the Optimizer to store values
        that are required and modified during each iteration.

        Here, we do not create any slots.

        :param var_list: list of variables
        """
        for v in var_list:
            self._zeros_slot(v, "momentum", self._name)

    def _apply_dense(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling.

        We simply add nodes that generate normally distributed noise here, one
        for each weight.
        The norm of the injected noise is placed into the TensorFlow summary.

        The discretization scheme is according to (1.59) in [dissertation Zofia Trstanova],
        i.e. 1st order Geometric Langevin Algorithm.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        friction_constant_t = math_ops.cast(self._friction_constant_t, var.dtype.base_dtype)
        step_width_t, inverse_temperature_t, random_noise_t = self._prepare_dense(grad, var)
        momentum = self.get_slot(var, "momentum")

        # \nabla V (q^n ) \Delta t
        scaled_gradient = step_width_t * grad

        with tf.variable_scope("accumulate", reuse=True):
            gradient_global = tf.get_variable("gradients")
            gradient_global_t = tf.assign_add(gradient_global, tf.reduce_sum(tf.multiply(scaled_gradient, scaled_gradient)))
            # configurational temperature
            virial_global = tf.get_variable("virials")
            virial_global_t = tf.assign_add(virial_global, tf.reduce_sum(tf.multiply(grad, var)))

        # 1/2 * p^{n}^t * p^{n}
        momentum_sq = 0.5 * tf.reduce_sum(tf.multiply(momentum, momentum))

        # p^{n+1} = p^{n} − \nabla V (q^n ) \Delta t
        momentum_full_step_t = momentum - scaled_gradient

        # q=^{n+1} = q^n + M^{-1} p_{n+1} ∆t
        var_update = state_ops.assign_add(var, step_width_t * momentum_full_step_t)

        alpha_t = tf.exp(-friction_constant_t * step_width_t)

        scaled_noise = tf.sqrt((1.-tf.pow(alpha_t, 2))/inverse_temperature_t) * random_noise_t
        with tf.variable_scope("accumulate", reuse=True):
            noise_global = tf.get_variable("noise")
            noise_global_t = tf.assign_add(noise_global, tf.pow(alpha_t, -2) * tf.reduce_sum(tf.multiply(scaled_noise, scaled_noise)))

        # p^{n+1} = \alpha_{\Delta t} p^{n+1} + \sqrt{ \frac{1-\alpha^2_{\Delta t}}{\beta} M } G^n
        momentum_t = momentum.assign(alpha_t * momentum_full_step_t + scaled_noise)
        with tf.variable_scope("accumulate", reuse=True):
            momentum_global = tf.get_variable("momenta")
            momentum_global_t = tf.assign_add(momentum_global, tf.reduce_sum(tf.multiply(momentum_t, momentum_t)))

        # as the loss evaluated with train_step is the "old" (not updated) loss, we
        # therefore also need to the use the old momentum for the kinetic energy
        with tf.variable_scope("accumulate", reuse=True):
            kinetic_energy = tf.get_variable("kinetic")
            kinetic_energy_t = tf.assign_add(kinetic_energy, momentum_sq)

        return control_flow_ops.group(*[gradient_global_t, virial_global_t, var_update,
                                        noise_global_t, momentum_t, momentum_global_t,
                                        kinetic_energy_t])

    def _apply_sparse(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of sparsely
        occupied tensors to perform the actual sampling.

        Note that this is not implemented so far.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        raise NotImplementedError("Sparse gradient updates are not supported.")
