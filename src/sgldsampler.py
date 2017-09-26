# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
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
        pass
    
    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        
        var_update = state_ops.assign_sub(var, lr_t * grad)
        return control_flow_ops.group(*[var_update])
