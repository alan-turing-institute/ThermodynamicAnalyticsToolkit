# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf


class SGLDSampler(optimizer.Optimizer):
    """ Implements a Stochastic Gradient Langevin Dynamics Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    We have the upgrade as: %\Delta\Theta = - \nabla \beta U(\Theta) \Delta t + \sqrt{\Delta t} N$,
    where $\beta$ is the inverse temperature coefficient, $\Delta t$ is the (discretization)
    step width and $\Theta$ is the parameter vector and $U(\Theta)$ the energy or loss function.
    """
    def __init__(self, step_width, inverse_temperature, seed=None, use_locking=False, name='SGLD'):
        """ Init function for this class.

        :param step_width: step width for gradient, also affects inject noise
        :param inverse_temperature: scale for gradients
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(SGLDSampler, self).__init__(use_locking, name)
        self._step_width = step_width
        self._seed = seed
        self.random_noise = None
        self.scaled_gradient = None
        self.scaled_noise = None
        self._inverse_temperature = inverse_temperature
    
    def _prepare(self):
        """ Converts step width into a tensor, if given as a floating-point
        number.
        """
        self._step_width_t = ops.convert_to_tensor(self._step_width, name="step_width")
        self._inverse_temperature_t = ops.convert_to_tensor(self._inverse_temperature, name="inverse_temperature")

    def _create_slots(self, var_list):
        """ Slots are internal resources for the Optimizer to store values
        that are required and modified during each iteration.

        Here, we do not create any slots.

        :param var_list: list of variables
        """
        pass

    def _prepare_dense(self, grad, var):
        """ Stuff common to all Langevin samplers.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: step_width, inverse_temperature, and noise tensors
        """
        step_width_t = math_ops.cast(self._step_width_t, var.dtype.base_dtype)
        inverse_temperature_t = math_ops.cast(self._inverse_temperature_t, var.dtype.base_dtype)
        #print("lr_t is "+str(self._lr))
        if self._seed is None:
            random_noise_t = tf.random_normal(grad.get_shape(), mean=0.,stddev=1.)
        else:
            random_noise_t = tf.random_normal(grad.get_shape(), mean=0., stddev=1., seed=self._seed)
        #print("random_noise_t has shape "+str(random_noise_t.get_shape())+" with seed "+str(self._seed))
        return step_width_t, inverse_temperature_t, random_noise_t

    def _apply_dense(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling.

        We simply add nodes that generate normally distributed noise here, one
        for each weight.
        The norm of the injected noise is placed into the TensorFlow summary.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        step_width_t, inverse_temperature_t, random_noise_t = self._prepare_dense(grad, var)

        # \nabla V (q^n ) \Delta t
        scaled_gradient = step_width_t * grad

        with tf.variable_scope("accumulate", reuse=True):
            gradient_global = tf.get_variable("gradients")
            gradient_global_t = tf.assign_add(gradient_global, tf.reduce_sum(tf.multiply(scaled_gradient, scaled_gradient)))

        scaled_noise = tf.sqrt(2.*step_width_t/inverse_temperature_t) * random_noise_t
        with tf.variable_scope("accumulate", reuse=True):
            noise_global = tf.get_variable("noise")
            noise_global_t = tf.assign_add(noise_global, tf.reduce_sum(tf.multiply(scaled_noise, scaled_noise)))

        var_update = state_ops.assign_sub(var, scaled_gradient + scaled_noise)
        return control_flow_ops.group(*[var_update, gradient_global_t, noise_global_t])

    def _apply_sparse(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of sparsely
        occupied tensors to perform the actual sampling.

        Note that this is not implemented so far.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        raise NotImplementedError("Sparse gradient updates are not supported.")
