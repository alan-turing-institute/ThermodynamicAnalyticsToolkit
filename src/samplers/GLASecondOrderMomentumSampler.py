# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
import tensorflow as tf

from DataDrivenSampler.samplers.GLAFirstOrderMomentumSampler import GLAFirstOrderMomentumSampler


class GLASecondOrderMomentumSampler(GLAFirstOrderMomentumSampler):
    """ Implements a Geometric Langevin Algorithm Momentum Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    """
    def __init__(self, step_width, inverse_temperature, friction_constant,
                 seed=None, use_locking=False, name='GLA_2ndOrder'):
        """ Init function for this class.

        :param step_width: step width for gradient, also affects inject noise
        :param inverse_temperature: scale for gradients
        :param friction_constant: scales the momenta
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(GLASecondOrderMomentumSampler, self).__init__(step_width, inverse_temperature,
                                                            friction_constant, seed, use_locking, name)

    def _apply_dense(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling.

        We simply add nodes that generate normally distributed noise here, one
        for each weight.
        The norm of the injected noise is placed into the TensorFlow summary.

        The discretization scheme is according to (1.59) in [dissertation Zofia Trstanova],
        i.e. 2nd order Geometric Langevin Algorithm.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        friction_constant_t = math_ops.cast(self._friction_constant_t, var.dtype.base_dtype)
        step_width_t, inverse_temperature_t, random_noise_t = self._prepare_dense(grad, var)

        print("Shape of grad "+str(grad.get_shape()))
        scaled_gradient = 0.5 * step_width_t * grad
        print("Shape of scaled_gradient "+str(scaled_gradient.get_shape()))
        self.scaled_gradient = tf.norm(grad)
        with tf.name_scope('sample'):
            tf.summary.scalar('scaled_gradient', self.scaled_gradient)

        momentum = self.get_slot(var, "momentum")

        # as the loss evaluated with train_step is the "old" (not updated) loss, we
        # therefore also need to the use the old momentum for the kinetic energy
        kinetic_energy_t = 0.5*tf.reduce_sum(tf.multiply(momentum, momentum))
        self.kinetic_energy = kinetic_energy_t
        with tf.name_scope('sample'):
            tf.summary.scalar('kinetic_energy', kinetic_energy_t)

        # p^{n+1/2} = p^{n} − \nabla V (q^n ) \Delta t/2
        momentum_half_step_t = momentum - scaled_gradient

        # q=^{n+1} = q^n + M^{-1} p_{n+1/2} ∆t
        var_update = state_ops.assign_add(var, step_width_t * momentum_half_step_t)

        # p^{n+1} = p^{n+1/2} − \nabla V (q^{n+1} ) \Delta t/2 (we use q^n here instead)
        momentum_full_step_t = momentum_half_step_t - scaled_gradient

        alpha_t = tf.exp(-friction_constant_t * step_width_t)

        scaled_noise = tf.sqrt((1.-tf.pow(alpha_t, 2))/inverse_temperature_t) * random_noise_t
        self.scaled_noise = tf.norm(scaled_noise)
        tf.summary.scalar('scaled_noise', self.scaled_noise)

        # p^{n+1} = \alpha_{\Delta t} p^{n+1} + \sqrt{ \frac{1-\alpha^2_{\Delta t}}{\beta} M } G^n
        momentum_update = alpha_t * momentum_full_step_t + scaled_noise
        momentum_t = momentum.assign(momentum_update)
        self.scaled_momentum = tf.norm(momentum_t)
        with tf.name_scope('sample'):
            tf.summary.scalar('scaled_momentum', self.scaled_momentum)

        return control_flow_ops.group(*[var_update, momentum_t])
