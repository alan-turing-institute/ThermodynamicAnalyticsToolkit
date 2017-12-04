from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow as tf


class GradientDescent(tf.train.GradientDescentOptimizer):
    """ We are extending TensorFlow's GradientDescentOptimizer to access the
    gradient's norm.
    """

    def __init__(self, learning_rate, use_locking=False, name='GradientDescent'):
        """ Init function to get access to learning rate.

        :param learning_rate:
        :param use_locking:
        :param name:
        """
        super(GradientDescent, self).__init__(learning_rate, use_locking, name)
        self._learning_rate = learning_rate
        self.scaled_gradient = None

    def _prepare(self):
        """ Convert internal learning_rate to proper tensor.
        """
        self._learning_rate_t = ops.convert_to_tensor(self._learning_rate, name="learning_rate")
        super(GradientDescent, self)._prepare()

    def _apply_dense(self, grad, var):
        """ Add scaled gradient and train as usual

        :param grad: gradients
        :param var: variables
        """
        lr_t = math_ops.cast(self._learning_rate_t, var.dtype.base_dtype)
        scaled_gradient = lr_t * grad
        with tf.variable_scope("accumulate", reuse=True):
            gradient_global = tf.get_variable("gradients", dtype=tf.float64)
            gradient_global_t = tf.assign_add(gradient_global, tf.reduce_sum(tf.multiply(scaled_gradient, scaled_gradient)))
            virial_global = tf.get_variable("virials", dtype=tf.float64)
            virial_global_t = tf.assign_add(virial_global, tf.reduce_sum(tf.multiply(grad, var)))
        control_group_gradient_descent_t = super(GradientDescent, self)._apply_dense(grad, var)
        return control_flow_ops.group(*[virial_global_t, control_group_gradient_descent_t, gradient_global_t])
