# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
import random

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class SGLDSampler(optimizer.Optimizer):
    ''' implements a Stochastic Gradient Langevin Dynamics Sampler
    in the form of a TensorFlow Optimizer
    '''
    def __init__(self, learning_rate, noise_scale=1., seed=None, use_locking=False, name='SGLDSampler'):
        super(SGLDSampler, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._seed = seed
        self.random_noise = None
        self._noise_scale = noise_scale
    
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    
    def _create_slots(self, var_list):
        pass

    def _apply_dense(self, grad, var):
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
        raise NotImplementedError("Sparse gradient updates are not supported.")
