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
    """
    def __init__(self, learning_rate, noise_scale=1., seed=None, use_locking=False, name='SGLDSampler'):
        """ Init function for this class.

        :param learning_rate: learning_rate, i.e. step width for optimizer to use
        :param noise_scale: scale of injected noise to gradients, 0. deactivates it
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(SGLDSampler, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._seed = seed
        self.random_noise = None
        self._noise_scale = noise_scale
    
    def _prepare(self):
        """ Converts learning rate into a tensor, if given as a floating-point
        number.
        """
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    
    def _create_slots(self, var_list):
        """ Slots are internal resources for the Optimizer to store values
        that are required and modified during each iteration.

        Here, we do not create any slots.

        :param var_list: list of variables
        """
        pass

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
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        #print("lr_t is "+str(self._lr))
        if self._seed is None:
            random_noise = tf.random_normal(grad.get_shape(), mean=0.,stddev=lr_t)
        else:
            random_noise = tf.random_normal(grad.get_shape(), mean=0., stddev=lr_t, seed=self._seed)
        #print("random_noise has shape "+str(random_noise.get_shape())+" with seed "+str(self._seed))
        self.random_noise = tf.norm(random_noise)
        tf.summary.scalar('noise', self.random_noise)

        var_update = state_ops.assign_sub(var,
                                          lr_t/2. * grad
                                          + tf.constant(self._noise_scale, tf.float32) * self.random_noise)
        return control_flow_ops.group(*[var_update])

    def _apply_sparse(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of sparsely
        occupied tensors to perform the actual sampling.

        Note that this is not implemented so far.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        raise NotImplementedError("Sparse gradient updates are not supported.")
