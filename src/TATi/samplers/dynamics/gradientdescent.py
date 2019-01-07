#
#    ThermodynamicAnalyticsToolkit - explore high-dimensional manifold of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
### 

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
import tensorflow as tf

from TATi.models.basetype import dds_basetype

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
        self.upper_boundary = None
        self.lower_boundary = None
        self.force_factor = .1
        self.force_power = 1.

    def _prepare(self):
        """ Convert internal learning_rate to proper tensor.
        """
        self._learning_rate_t = ops.convert_to_tensor(self._learning_rate, name="learning_rate")
        super(GradientDescent, self)._prepare()

    def set_prior(self, prior):
        """ Sets the parameters for enforcing a prior.

        :param prior: dict with keys factor, lower_boundary and upper_boundary that
                specifies a wall-repelling force to ensure a prior on the parameters
        """
        if "factor" in prior:
            self.force_factor = prior["factor"]
        if "lower_boundary" in prior:
            self.lower_boundary = prior["lower_boundary"]
        if "power" in prior:
            self.force_power = prior["power"]
        if "upper_boundary" in prior:
            self.upper_boundary = prior["upper_boundary"]

    def _apply_prior(self, var):
        """ Returns two prior constraining force nodes that have linearly increasing
        strength with distance to wall (and beyond).

        Note that force is continuous,
        not smooth itself!

        The domain is specified by the interval [upper_boundary, lower_boundary].

        Cases:
        ------
          1. both ub and lb specified, then in 0.01 of domain length (ub-lb) the force
             begins
          2. if only ub is specified, then within 0.01 of domain length (ub) ...
          3. if only lb is specified as it negative, then within -0.01 ...
             otherwise we use a fixed relative domain length of 0.01

        :param var:
        :return:
        """

        def wall_force(signed_distance, wall_size):
            return -1. * self.force_factor * tf.pow(
                tf.abs(tf.minimum(tf.div(signed_distance, wall_size), 1.0)-1.0),
                self.force_power)

        def tether_force(signed_distance):
            return self.force_factor * tf.pow(
                tf.maximum(signed_distance, 0.0),
                self.force_power)

        # mind that sign needs to be reversed as we update using _negative_ gradient
        if self.upper_boundary is not None or self.lower_boundary is not None:
            if self.upper_boundary is not None and self.lower_boundary is not None:
                # add prior for a fixed domain with repelling walls
                ub = tf.constant(self.upper_boundary, shape=var.shape, dtype=var.dtype.base_dtype)
                lb = tf.constant(self.lower_boundary, shape=var.shape, dtype=var.dtype.base_dtype)
                domain_length = self.upper_boundary - self.lower_boundary
                if abs(domain_length) > 1e-10:
                    wall_size = tf.constant(domain_length * 0.01, shape=var.shape, dtype=var.dtype.base_dtype)
                    # force linearly increase "wall_size" inwards away from wall
                    ub_repell = -1. * wall_force(ub-var, wall_size)
                    lb_repell = wall_force(var-lb, wall_size)
                else:
                    # fixed to a central point
                    ub_repell = -1. * tether_force(ub-var)
                    lb_repell = tether_force(var-lb)
            else:
                if self.upper_boundary is not None:
                    ub = tf.constant(self.upper_boundary, shape=var.shape, dtype=var.dtype.base_dtype)
                    if self.upper_boundary < 0:
                        domain_length = -1.*self.upper_boundary
                    else:
                        domain_length = 1.
                    wall_size = tf.constant(domain_length * 0.01, shape=var.shape, dtype=var.dtype.base_dtype)
                    ub_repell = -1. * wall_force(ub-var, wall_size)
                    lb_repell = tf.zeros(shape=var.shape, dtype=var.dtype.base_dtype)
                else:
                    lb = tf.constant(self.lower_boundary, shape=var.shape, dtype=var.dtype.base_dtype)
                    if self.lower_boundary < 0:
                        domain_length = -1.*self.lower_boundary
                    else:
                        domain_length = 1.
                    wall_size = tf.constant(domain_length * 0.01, shape=var.shape, dtype=var.dtype.base_dtype)
                    ub_repell = tf.zeros(shape=var.shape, dtype=var.dtype.base_dtype)
                    lb_repell = wall_force(var - lb, wall_size)
        else:
            ub_repell = tf.zeros(shape=var.shape, dtype=var.dtype.base_dtype)
            lb_repell = tf.zeros(shape=var.shape, dtype=var.dtype.base_dtype)
        return ub_repell, lb_repell

    def _apply_dense(self, grad, var):
        """ Add scaled gradient and train as usual

        :param grad: gradients
        :param var: variables
        """
        lr_t = math_ops.cast(self._learning_rate_t, var.dtype.base_dtype)
        scaled_gradient = lr_t * grad
        with tf.variable_scope("accumulate", reuse=True):
            gradient_global = tf.get_variable("gradients", dtype=dds_basetype)
            gradient_global_t = tf.assign_add(gradient_global, tf.reduce_sum(tf.multiply(scaled_gradient, scaled_gradient)))
            virial_global = tf.get_variable("virials", dtype=dds_basetype)
            virial_global_t = tf.assign_add(virial_global, tf.reduce_sum(tf.multiply(grad, var)))

        # prior force act directly on var
        ub_repell, lb_repell = self._apply_prior(var)
        prior_force = (ub_repell + lb_repell)

        # make sure virial and gradients are evaluated before we update variables
        with tf.control_dependencies([virial_global_t, gradient_global_t]):
            control_group_gradient_descent_t = super(GradientDescent, self)._apply_dense(grad+prior_force, var)

        # note: these are evaluated in any order, use control_dependencies if required
        return control_flow_ops.group(*[virial_global_t, control_group_gradient_descent_t, gradient_global_t])
