# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

from sgldsampler import SGLDSampler


class SGLDMomentumSampler(SGLDSampler):
    """ Implements a Stochastic Gradient Langevin Dynamics Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    We have the upgrade as: %\Delta\Theta = - \nabla \beta U(\Theta) \Delta t + \sqrt{\Delta t} N$,
    where $\beta$ is the inverse temperature coefficient, $\Delta t$ is the (discretization)
    step width and $\Theta$ is the parameter vector and $U(\Theta)$ the energy or loss function.
    """
    def __init__(self, step_width, inverse_temperature, friction_constant,
                 seed=None, use_locking=False, name='StochasticMomentumLangevin'):
        """ Init function for this class.

        :param step_width: step width for gradient, also affects inject noise
        :param inverse_temperature: scale for gradients
        :param friction_constant: scales the momenta
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(SGLDMomentumSampler, self).__init__(step_width, inverse_temperature,
                                                  seed, use_locking, name)
        self._friction_constant = friction_constant
        self.scaled_momentum = None
        self.momentum = None

    def _prepare(self):
        """ Converts step width into a tensor, if given as a floating-point
        number.
        """
        super(SGLDMomentumSampler, self)._prepare()
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

        The formulas here are as in [Stoltz, Trstanova, 2016].

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        friction_constant_t = math_ops.cast(self._friction_constant_t, var.dtype.base_dtype)
        step_width_t, inverse_temperature_t, random_noise_t = self._prepare_dense(grad, var)

        scaled_noise = tf.sqrt(2.*friction_constant_t/inverse_temperature_t) * random_noise_t
        self.scaled_noise = tf.norm(scaled_noise)
        tf.summary.scalar('scaled_noise', self.scaled_noise)

        momentum = self.get_slot(var, "momentum")
        scaled_momentum = 2. * friction_constant_t * momentum * step_width_t
        self.scaled_momentum = tf.norm(scaled_momentum)
        tf.summary.scalar('scaled_momentum', self.scaled_momentum)

        scaled_gradient = step_width_t * grad
        self.scaled_gradient = tf.norm(scaled_gradient)
        tf.summary.scalar('scaled_gradient', self.scaled_gradient)

        momentum_update = momentum + scaled_gradient + scaled_noise
        momentum_t = momentum.assign(momentum_update)
        self.momentum = tf.norm(momentum_t)
        tf.summary.scalar('momentum', self.momentum)

        var_update = state_ops.assign_sub(var, momentum_update + scaled_gradient)
        return control_flow_ops.group(*[var_update, momentum_t])

    def _apply_sparse(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of sparsely
        occupied tensors to perform the actual sampling.

        Note that this is not implemented so far.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        raise NotImplementedError("Sparse gradient updates are not supported.")
