# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import numpy as np

class SGLDSampler(optimizer.Optimizer):
    ''' implements a Stochastic Gradient Langevin Dynamics Sampler
    in the form of a TensorFlow Optimizer
    '''
    def __init__(self, learning_rate, use_locking=False, name='SGLDSampler'):
        super(SGLDSampler, self).__init__(use_locking, name)
        self._lr = learning_rate
    
    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    
    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "noise", self._name)
    
    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        #print("lr_t is "+str(self._lr))
        noise = self.get_slot(var, "noise")
        print("noise has shape "+str(noise.get_shape()))
        no_vars = np.cumprod(noise.get_shape())[-1]
        random_noise = tf.reshape(
            math_ops.cast(np.random.normal(loc=0,scale=self._lr, size=no_vars),
                          var.dtype.base_dtype), noise.get_shape())
        print("random_noise has shape "+str(random_noise.get_shape()))        
        noise_t = noise.assign(random_noise)
        
        var_update = state_ops.assign_sub(var, lr_t * grad + noise_t)
        return control_flow_ops.group(*[var_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")