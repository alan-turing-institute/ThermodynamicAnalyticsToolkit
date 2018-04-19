from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

from TATi.models.basetype import dds_basetype


class ReplicaOptimizer(optimizer.Optimizer):
    """ This implements a version of the optimizer that has more than one replica
    that exchange their gradients among one another.
    """
    def __init__(self, use_locking=False, name='ReplicaOptimizer'):
        """ Init function for this class.

        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(ReplicaOptimizer, self).__init__(use_locking, name)

    def compute_and_check_gradients(self, loss, var_list=None,
                 gate_gradients=optimizer.Optimizer.GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, grad_loss=None):
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops" \
                +" that do not support gradients, between variables %s and loss %s." \
                % ([str(v) for _, v in grads_and_vars], loss))

        return grads_and_vars
