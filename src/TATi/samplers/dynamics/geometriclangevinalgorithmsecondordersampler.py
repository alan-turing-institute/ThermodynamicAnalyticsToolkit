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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

from TATi.models.basetype import dds_basetype
from TATi.samplers.dynamics.geometriclangevinalgorithmfirstordersampler import \
    GeometricLangevinAlgorithmFirstOrderSampler


class GeometricLangevinAlgorithmSecondOrderSampler(GeometricLangevinAlgorithmFirstOrderSampler):
    """Implements a Geometric Langevin Algorithm Momentum Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    Args:

    Returns:

    """
    def __init__(self, calculate_accumulates, covariance_blending, step_width,
                 inverse_temperature, friction_constant,
                 seed=None, use_locking=False, name='GLA_2ndOrder'):
        """Init function for this class.

        Args:
          calculate_accumulates: whether accumulates (gradient norm, noise, norm, kinetic energy, ...) are calculated
            every step (extra work but required for run info dataframe/file and averages dataframe/file)
          covariance_blending: covariance identity blending value eta to use in creating the preconditioning matrix
          step_width: step width for gradient, also affects inject noise
          inverse_temperature: scale for gradients
          friction_constant: scales the momenta
          seed: seed value of the random number generator for generating reproducible runs (Default value = None)
          use_locking: whether to lock in the context of multi-threaded operations (Default value = False)
          name: internal name of optimizer (Default value = 'GLA_2ndOrder')

        Returns:

        """
        super(GeometricLangevinAlgorithmSecondOrderSampler, self).__init__(
            calculate_accumulates, covariance_blending, step_width,
            inverse_temperature, friction_constant, seed, use_locking, name)


    def _apply_dense(self, grads_and_vars, var):
        """Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling.
        
        We simply add nodes that generate normally distributed noise here, one
        for each weight.
        The norm of the injected noise is placed into the TensorFlow summary.
        
        The discretization scheme is according to (1.59) in [dissertation Zofia Trstanova],
        i.e. 2nd order Geometric Langevin Algorithm.

        Note:
          This is "BABO" in the notation of [Leimkuhler, 2012] which is permuted
          to become BOBA as tensorflow's Optimizer class requires gradient evaluation
          to occur at the beginning.

        Args:
          grads_and_vars: gradient nodes over all walkers and all variables
          var: parameters of the neural network

        Returns:
          a group of operations to be added to the graph

        """
        precondition_matrix, grad = self._pick_grad(grads_and_vars, var)
        friction_constant_t = math_ops.cast(self._friction_constant_t, var.dtype.base_dtype)
        step_width_t, inverse_temperature_t, random_noise_t = self._prepare_dense(grad, var)

        # p^{n}
        momentum = self.get_slot(var, "momentum")

        # \nabla V (q^n ) \Delta t + prior constraining force
        scaled_gradient = step_width_t * grad

        # conditionally compute accumulates
        gradient_global_t = self._get_accumulate_conditional("gradients",
            lambda: self._accumulate_norm("gradients", scaled_gradient))
        virial_global_t = self._get_accumulate_conditional("virials",
            lambda: self._accumulate_scalar_product("virials", grad, var))

        # p^{n+1/2} = p^{n} − \nabla V (q^n ) \Delta t/2
        momentum_half_step_t = momentum - 0.5 * scaled_gradient

        alpha_t = tf.exp(-friction_constant_t * step_width_t)

        scaled_noise = tf.sqrt((1.-tf.pow(alpha_t, 2))/inverse_temperature_t) * random_noise_t
        rescaled_noise = scaled_noise/alpha_t
        noise_global_t = self._get_accumulate_conditional("noise",
            lambda: self._accumulate_norm("noise", rescaled_noise))
        momentum_noise_step_t = alpha_t * momentum_half_step_t + scaled_noise

        # 1/2 * p^{n}^t * p^{n}
        # as the loss evaluated with train_step is the "old" (not updated) loss, we
        # therefore also need to the use the old momentum for the kinetic energy
        # However, "B" and "O" from the last step still need to be done
        kinetic_energy_t = self._get_accumulate_conditional("kinetic_energy",
            lambda: self._accumulate_norm("kinetic_energy", momentum_noise_step_t, 0.5))

        # p^{n+1} = p^{n+1/2} − \nabla V (q^{n+1} ) \Delta t/2
        with tf.control_dependencies([kinetic_energy_t]):
            momentum_t = momentum.assign(momentum_noise_step_t - 0.5 * scaled_gradient)

        # prior force act directly on var
        ub_repell, lb_repell = self._apply_prior(var)
        prior_force = step_width_t * (ub_repell + lb_repell)

        if len(grads_and_vars) != 1:
            preconditioned_momentum_t = tf.reshape(
                tf.matmul(tf.expand_dims(tf.reshape(momentum_t, [-1]), 0), precondition_matrix),
                var.shape)
        else:
            preconditioned_momentum_t = momentum_t

        # p^{n+1} = \alpha_{\Delta t} p^{n+1} + \sqrt{ \frac{1-\alpha^2_{\Delta t}}{\beta} M } G^n
        momentum_global_t = self._get_accumulate_conditional("momenta",
            lambda: self._accumulate_norm("momenta", momentum_t))
        inertia_global_t = self._get_accumulate_conditional("inertia",
            lambda: self._accumulate_scalar_product("inertia", momentum_t, var))

        # make sure virial is evaluated before we update variables
        with tf.control_dependencies([virial_global_t, inertia_global_t]):
            # q=^{n+1} = q^n + M^{-1} p_{n+1/2} ∆t
            var_update = state_ops.assign_add(var, step_width_t * preconditioned_momentum_t - prior_force)

        # note: these are evaluated in any order, use control_dependencies if required
        return control_flow_ops.group(*[noise_global_t, gradient_global_t,
                                        virial_global_t, inertia_global_t, kinetic_energy_t,
                                        var_update,
                                        momentum_t, momentum_global_t,
                                        ])
