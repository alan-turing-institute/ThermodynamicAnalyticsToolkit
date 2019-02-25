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
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

from TATi.models.basetype import dds_basetype
from TATi.samplers.dynamics.geometriclangevinalgorithmsecondordersampler \
    import GeometricLangevinAlgorithmSecondOrderSampler


class CovarianceControlledAdaptiveLangevinThermostat(GeometricLangevinAlgorithmSecondOrderSampler):
    """Implements a Geometric Langevin Algorithm Momentum Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    Args:

    Returns:

    """
    def __init__(self, calculate_accumulates, covariance_blending, step_width,
                 inverse_temperature, friction_constant, sigma, sigmaA,
                 seed=None, use_locking=False, name='CovarianceControlledAdaptiveLangevinThermostat'):
        """Init function for this class.

        Args:
          calculate_accumulates: whether accumulates (gradient norm, noise, norm, kinetic energy, ...) are calculated
            every step (extra work but required for run info dataframe/file and averages dataframe/file)
          covariance_blending: covariance identity blending value eta to use in creating the preconditioning matrix
          step_width: step width for gradient, also affects inject noise
          inverse_temperature: scale for gradients
          friction_constant: scales the momenta
          sigma: scale of noise injected to momentum per step
          sigmaA: scale of noise in convex combination
          seed: seed value of the random number generator for generating reproducible runs (Default value = None)
          use_locking: whether to lock in the context of multi-threaded operations (Default value = False)
          name: internal name of optimizer (Default value = 'CovarianceControlledAdaptiveLangevinThermostat')

        Returns:

        """
        super(CovarianceControlledAdaptiveLangevinThermostat, self).__init__(
            calculate_accumulates, covariance_blending, step_width,
            inverse_temperature, friction_constant, seed, use_locking, name)
        self._sigma = sigma
        self._sigmaA = sigmaA

    def _prepare(self):
        """Converts step width into a tensor, if given as a floating-point
        number.

        Args:

        Returns:

        """
        super(CovarianceControlledAdaptiveLangevinThermostat, self)._prepare()
        self._sigma_t = ops.convert_to_tensor(self._sigma, name="sigma")
        self._sigmaA_t = ops.convert_to_tensor(self._sigmaA, name="sigmaA")
        with tf.variable_scope("CCAdL"):
            gammaAdapt = tf.get_variable("gammaAdapt", shape=[], trainable=False,
                                         initializer=tf.zeros_initializer,
                                         use_resource=True, dtype=dds_basetype)
            total_noise = tf.get_variable("total_noise", shape=[], trainable=False,
                                          initializer=tf.zeros_initializer,
                                          use_resource=True, dtype=dds_basetype)

    def _apply_dense(self, grads_and_vars, var):
        """Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling.
        
        We simply add nodes that generate normally distributed noise here, one
        for each weight.
        The norm of the injected noise is placed into the TensorFlow summary.
        
        The discretization scheme is according to (1.59) in [dissertation Zofia Trstanova],
        i.e. 2nd order Geometric Langevin Algorithm.

        Args:
          grads_and_vars: gradient nodes over all walkers and all variables
          var: parameters of the neural network

        Returns:
          a group of operations to be added to the graph

        """
        # get number of parameters (for this layer)
        precondition_matrix, grad = self._pick_grad(grads_and_vars, var)
        dim = math_ops.cast(tf.size(var), dds_basetype)

        sigma_t = math_ops.cast(self._sigma_t, var.dtype.base_dtype)
        sigmaA_t = math_ops.cast(self._sigmaA_t, var.dtype.base_dtype)
        step_width_t, inverse_temperature_t, random_noise_t = self._prepare_dense(grad, var)
        if self._seed is not None:
            sigma_random_noise_t = tf.random_normal(grad.get_shape(), mean=0., stddev=1., seed=self._seed+1, dtype=dds_basetype)
            sigmaA_random_noise_t = tf.random_normal(grad.get_shape(), mean=0., stddev=1., seed=self._seed+2, dtype=dds_basetype)
        else:
            sigma_random_noise_t = tf.random_normal(grad.get_shape(), mean=0., stddev=1., dtype=dds_basetype)
            sigmaA_random_noise_t = tf.random_normal(grad.get_shape(), mean=0., stddev=1., dtype=dds_basetype)

        # sigma * np.random.rand(p.shape[0]) * 0.5 * step_width_t
        scaled_noise = sigma_t * sigma_random_noise_t * 0.5 * step_width_t
        scaled_noiseA = sigmaA_t * sigmaA_random_noise_t * tf.sqrt(step_width_t)
        with tf.control_dependencies([scaled_noise]):
            with tf.variable_scope("CCAdL", reuse=True):
                total_noise = tf.get_variable("total_noise", dtype=dds_basetype)
                total_noise_init_t = total_noise.assign(2.0 * tf.reduce_sum(tf.multiply(scaled_noise,scaled_noise)))

        # p^{n}
        momentum = self.get_slot(var, "momentum")

        scaled_gradient = step_width_t * grad

        gradient_global_t = self._get_accumulate_conditional("gradients",
            lambda: self._accumulate_norm("gradients", scaled_gradient))
        virial_global_t = self._get_accumulate_conditional("virials",
            lambda: self._accumulate_scalar_product("virials", grad, var))

        # p = step.B(p, f, 0.5 * self.dt)
        momentum_final_step_t = momentum - 0.5 * scaled_gradient

        inertia_global_t = self._get_accumulate_conditional("inertia",
            lambda: self._accumulate_scalar_product("inertia", momentum_final_step_t, var))

        # 1/2 * p^{n}^t * p^{n}
        momentum_sq = 0.5 * tf.reduce_sum(tf.multiply(momentum_final_step_t, momentum_final_step_t))

        # as the loss evaluated with train_step is the "old" (not updated) loss, we
        # therefore also need to the use the old momentum for the kinetic energy
        kinetic_energy_t = self._get_accumulate_conditional("kinetic_energy",
            lambda: self._accumulate_value("kinetic_energy", momentum_sq, 0.5))
        momentum_global_t = self._get_accumulate_conditional("momenta",
            lambda: self._accumulate_value("momenta", momentum_sq))

        #p = step.B(p, f, 0.5 * self.dt) + sigma * np.random.rand(p.shape[0]) * 0.5 * step_width_t
        momentum_half_step_t = momentum_final_step_t - 0.5 * scaled_gradient
        momentum_half_step_plus_noise_t = momentum_half_step_t + scaled_noise

        if len(grads_and_vars) != 1:
            preconditioned_momentum_half_step_plus_noise_t = tf.reshape(
                tf.matmul(tf.expand_dims(tf.reshape(momentum_half_step_plus_noise_t, [-1]), 0), precondition_matrix),
                var.shape)
        else:
            preconditioned_momentum_half_step_plus_noise_t = momentum_half_step_plus_noise_t

        #x = step.A(x, p, 0.5 * self.dt, 1.)
        with tf.control_dependencies([virial_global_t, inertia_global_t]):
            var_update_half_step_t = state_ops.assign_add(
                var, 0.5 * step_width_t * preconditioned_momentum_half_step_plus_noise_t)

        #gammaAdapt = gammaAdapt + (np.dot(p, p) - self.dim * self.T) * 0.5 * self.dt
        with tf.control_dependencies([momentum_half_step_plus_noise_t]):
            half_temperature_difference =  0.5 * (tf.reduce_sum(tf.multiply(momentum_half_step_plus_noise_t, momentum_half_step_plus_noise_t)) \
                                                  - dim /  inverse_temperature_t)
        with tf.variable_scope("CCAdL", reuse=True):
            gammaAdapt = tf.get_variable("gammaAdapt", dtype=dds_basetype)
            gammaAdapt_half_step_t = state_ops.assign_add(gammaAdapt,
                                                          half_temperature_difference * step_width_t)

        # see https://stackoverflow.com/questions/37063952/confused-by-the-behavior-of-tf-cond
        # In other words, each branch inside a tf.cond is evaluated. All "side effects"
        # need to be hidden away inside tf.control_dependencies.
        # I.E. DONT PLACE INSIDE NODES (confusing indeed)
        def accept_block():
            # p = p +sigmaA* np.random.randn(p.shape[0])*np.sqrt(self.mass*self.dt)
            with tf.control_dependencies([momentum.assign(
                            momentum_half_step_plus_noise_t \
                            + scaled_noiseA), \
                    total_noise.assign(total_noise_init_t + tf.reduce_sum(tf.multiply(scaled_noiseA,scaled_noiseA)))]):
                return tf.identity(momentum)

        # alpha= np.exp(-gammaAdapt* self.dt)
        alpha = tf.exp(-gammaAdapt_half_step_t * step_width_t)
        new_noise = sigmaA_t * tf.sqrt((1.0 - alpha * alpha) / (2.0 * gammaAdapt_half_step_t)) * sigmaA_random_noise_t
        def reject_block():
            # p = alpha*p + sigmaA*np.sqrt(self.mass* (1.0-alpha*alpha)/(2.0*gammaAdapt))*np.random.randn(p.shape[0])
            with tf.control_dependencies([momentum.assign(alpha * momentum_half_step_plus_noise_t \
                                                          + new_noise),
                                          total_noise.assign(total_noise_init_t + tf.reduce_sum(tf.pow(alpha, -2)* tf.multiply(new_noise,new_noise)))]):
                return tf.identity(momentum)

        # if (np.abs(gammaAdapt) < 0.1):
        with tf.control_dependencies([total_noise_init_t, momentum_half_step_plus_noise_t, scaled_noiseA, new_noise]):
            if_block_t = tf.cond(
                tf.less(tf.abs(gammaAdapt_half_step_t), tf.constant(0.1, dtype=dds_basetype)),
                accept_block,
                reject_block)

        with tf.control_dependencies([if_block_t]):
           updated_kinetic_energy_t = tf.reduce_sum(tf.multiply(momentum, momentum))
           full_temperature_difference = 0.5 * (updated_kinetic_energy_t - dim / inverse_temperature_t)

        #gammaAdapt = gammaAdapt + (np.dot(p, p) - self.dim * self.T) * 0.5 * self.dt
        with tf.variable_scope("CCAdL", reuse=True):
            gammaAdapt_full_step = state_ops.assign_add(gammaAdapt, full_temperature_difference * step_width_t)

        # prior force act directly on var
        ub_repell, lb_repell = self._apply_prior(var)
        prior_force = step_width_t * (ub_repell + lb_repell)

        if len(grads_and_vars) != 1:
            preconditioned_momentum = tf.reshape(
                tf.matmul(tf.expand_dims(tf.reshape(momentum, [-1]), 0), precondition_matrix),
                var.shape)
        else:
            preconditioned_momentum = momentum

        #x = step.A(x, p, 0.5 * self.dt, 1.)
        with tf.control_dependencies([if_block_t]):
            var_update_full_step_t = state_ops.assign_add(
                var, 0.5 * step_width_t * preconditioned_momentum - prior_force)

        # we split last B step up into adding noise (such that noise_global_t
        # is complete for this step) and the updated momentum which is done in
        # the very first lines of the next step
        # p = p + sigma * np.random.randn(p.shape[0]) * 0.5 * self.dt
        momentum_final_step_plus_noise_t = momentum_final_step_t + scaled_noise

        def _accumulate_total_noise():
            with tf.control_dependencies([if_block_t, total_noise]):
                with tf.variable_scope("accumulate", reuse=True):
                    noise_global = tf.get_variable("noise", dtype=dds_basetype)
                    with tf.control_dependencies([noise_global.assign_add(tf.reduce_sum(total_noise))]):
                        return tf.identity(noise_global)

        # conditionally calculate norm of noise
        noise_global_t = self._get_accumulate_conditional("noise", _accumulate_total_noise)

        # NOTE: adhere to ordering of node evaluation here!
        return control_flow_ops.group(*[virial_global_t, inertia_global_t, gradient_global_t,
                                        momentum_final_step_t, momentum_global_t,
                                        kinetic_energy_t, momentum_half_step_plus_noise_t,
                                        var_update_half_step_t, gammaAdapt_half_step_t,
                                        if_block_t, gammaAdapt_full_step,
                                        var_update_full_step_t, momentum_final_step_plus_noise_t,
                                        noise_global_t])
