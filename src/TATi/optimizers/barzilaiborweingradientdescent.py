#
#    ThermodynamicAnalyticsToolkit - analyze loss manifolds of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#    The TATi authors, see file AUTHORS, have asserted their moral rights.
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

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from TATi.optimizers.gradientdescent import GradientDescent
from TATi.models.basetype import dds_basetype

class BarzilaiBorweinGradientDescent(GradientDescent):
    """ This implements a gradient descent where the learning rate is
    automatically obtained through Barzilai-Borwein.

    """
    def __init__(self, calculate_accumulates, learning_rate, use_locking=False, name='BarzilaiBorweinGradientDescent'):
        """Init function to get access to learning rate.

        Args:
          calculate_accumulates: whether accumulates (gradient norm, noise, norm, kinetic energy, ...) are calculated
            every step (extra work but required for run info dataframe/file and averages dataframe/file)
          learning_rate: learning rate scales the gradient
          name:  name of optimizer method (Default value = 'GradientDescent')
          use_locking:  (Default value = False)

        Returns:

        """
        super(BarzilaiBorweinGradientDescent, self).__init__(calculate_accumulates,
                                                             learning_rate, use_locking, name)

    def _get_BarzilaiBorweinStepwidth(self, lr_current, position_difference, gradient_difference):
        # TODO: this needs to be converted into tf lingo
        norm_grad_difference = tf.norm(gradient_difference)

        BB_stepwidth = tf.reduce_sum(tf.multiply(position_difference, gradient_difference)) /\
                       (norm_grad_difference * norm_grad_difference)

        def BB_rate():
            with tf.control_dependencies([
                lr_current.assign(BB_stepwidth*tf.sign(BB_stepwidth))
            ]):
                return tf.identity(lr_current)

        def BB_pass():
            return tf.identity(lr_current)

        def BB_reset():
            with tf.control_dependencies([
                lr_current.assign(self._learning_rate_t)
            ]):
                return tf.identity(lr_current)

        def BB_upper_bound():
            return tf.constant(1.)

        step_calc_cond = tf.Print(
            tf.cond(
            tf.greater(norm_grad_difference, 1e-13),
            BB_rate, BB_pass),
            [norm_grad_difference], "norm_grad_difference")

        check_too_small_cond = tf.Print(
            tf.cond(
            tf.less(step_calc_cond, 1e-10),
            BB_reset, BB_pass),
            [step_calc_cond], "step after calc:")

        check_too_large_cond = tf.cond(
            tf.less(check_too_small_cond, 1e+10),
            BB_pass, BB_upper_bound)

        return tf.Print(check_too_large_cond,
                        [check_too_large_cond], "final step width")

    def _create_slots(self, var_list):
        """Slots are internal resources for the Optimizer to store values
        that are required and modified during each iteration.

        Here, we do not create any slots.

        Args:
          var_list: list of variables

        Returns:

        """
        for v in var_list:
            self._zeros_slot(v, "old_position", self._name)
            self._zeros_slot(v, "old_gradient", self._name)

    def _get_learning_rate(self, grad, var):
        # get slots for old grad and var
        old_var = self.get_slot(var, "old_position")
        old_grad = self.get_slot(var, "old_gradient")

        lr_current = tf.Variable(0., trainable=False, dtype=var.dtype.base_dtype)

        # get step width
        position_difference = tf.Print(var - old_var, [old_var, var], "old and new var:")
        gradient_difference = tf.Print(grad - old_grad, [old_grad, grad], "old and new grad:")
        lr_update = self._get_BarzilaiBorweinStepwidth(
            lr_current,
            tf.Print(position_difference, [position_difference], "var diff:"),
            tf.Print(gradient_difference, [gradient_difference], "grad diff:"))

        # update old_var and old_grad
        with tf.control_dependencies([lr_update]):
            update_old_var = old_var.assign(var)
            update_old_grad = old_grad.assign(grad)

        with tf.control_dependencies([update_old_var, update_old_grad]):
            lr_current_assign = lr_current.assign(lr_update)

        return lr_current_assign
