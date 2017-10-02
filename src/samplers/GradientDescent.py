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
        self.scaled_gradient = tf.norm(lr_t * grad)
        tf.summary.scalar('scaled_gradient', self.scaled_gradient)
        return super(GradientDescent, self)._apply_dense(grad, var)