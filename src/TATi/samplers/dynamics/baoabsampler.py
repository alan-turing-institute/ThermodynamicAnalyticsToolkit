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


class BAOABSampler(GeometricLangevinAlgorithmFirstOrderSampler):
    """Implements a Geometric Langevin Algorithm Momentum Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    Args:

    Returns:

    """
    def __init__(self, calculate_accumulates, covariance_blending,
                 step_width, inverse_temperature, friction_constant,
                 seed=None, use_locking=False, name='BAOAB'):
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
          name: internal name of optimizer (Default value = 'BAOAB')

        Returns:

        """
        super(BAOABSampler, self).__init__(calculate_accumulates, covariance_blending, step_width,
                                           inverse_temperature, friction_constant,
                                           seed, use_locking, name)


    '''
    legend:
        qn: position at step n
        half_qn: position at step n+1/2
        pn: momentum at step n
        half_pn: momentum at step n+1/2
        gn: gradient w.r.t. pn
        half_gn: gradient w.r.t. half_pn
        rn(i): noise in step n (first, second call)
        alpha(i): exp(-h/i gamma)
        zeta(i): [k_B T (1-exp(-i h gamma)]**(1/2)
        
    ABAO:
        half_qn = A(qn, pn, h/2)
        --- force calc
        half_pn = B(pn, half_gn, h)
        next_qn = A(half_qn, half_pn, h/2)
        next_pn = O(half_pn, rn, alpha1, zeta2)
        --- calc kinetic energy, calc loss
        
    BABO:
        half_pn = B(pn, gn, h/2)
        next_qn = A(qn, half_pn, h)
        --- force calc
        tilde_half_pn = B(half_pn, next_gn, h/2)
        next_pn = O(tilde_half_pn, rn, alpha1, zeta2)
        --- calc kinetic energy, calc loss
        
    ABOBA:
        half_qn = A(qn, pn, h/2)
        --- force calc
        half_pn = B(pn, half_gn, h)
        tilde_half_pn = O(half_pn, rn, alpha1, zeta2)
        next_pn = B(tilde_half_pn, half_gn, h/2)
        next_qn = A(half_qn, next_pn, h/2)        
        --- calc kinetic energy, calc loss

    BAOAB:
        half_pn = B(pn, gn, h/2)
        half_qn = A(qn, half_pn, h/2)
        tilde_half_pn = O(half_pn, rn, alpha1, zeta2)
        next_qn = A(half_qn, tilde_half_pn, h/2)
        --- force calc
        next_pn = B(tilde_half_pn, next_gn, h/2)
        --- calc kinetic energy, calc loss
        
    OABAO:
        half_pn = O(pn, rn1, alpha2, zeta1)
        half_qn = A(qn, half_pn, h/2)
        --- force calc
        tilde_half_pn = B(half_pn, half_gn, h)
        next_qn = A(half_qn, tilde_half_pn, h/2)
        next_pn = O(tilde_half_pn, rn2, alpha2, zeta1)
        --- calc kinetic energy, calc loss
        
    OBABO:
        half_pn = O(pn, rn1, alpha2, zeta1)
        tilde_half_pn = B(half_pn, gn, h/2)
        next_qn = A(qn, tilde_half_pn, h)
        --- force calc
        tilde_next_pn = B(tilde_half_pn, next_gn, h/2)
        next_pn = O(tilde_next_pn, rn2, alpha2, zeta1)
        --- calc kinetic energy, calc loss

    Let us shift the steps such that force calc is always at the beginning
    
    ABAO:
        --- force calc
        half_pn = B(pn, half_gn, h)
        next_qn = A(half_qn, half_pn, h/2)
        next_pn = O(half_pn, rn, alpha1, zeta2)
        --- calc kinetic energy, calc loss: next_qn -> qn, next_pn -> pn, next_gn -> gn
        half_qn = A(qn, pn, h/2)

        BAD: weights are half-updated
        
    BABO:
        --- force calc
        tilde_half_pn = B(half_pn, next_gn, h/2)
        next_pn = O(tilde_half_pn, rn, alpha1, zeta2)
        --- calc kinetic energy, calc loss: next_qn -> qn, next_pn -> pn, next_gn -> gn
        half_pn = B(pn, gn, h/2)
        next_qn = A(qn, half_pn, h)
        
        BAD: momenta are half-updated
        
    ABOBA:
        --- force calc
        half_pn = B(pn, half_gn, h)
        tilde_half_pn = O(half_pn, rn, alpha1, zeta2)
        next_pn = B(tilde_half_pn, half_gn, h/2)
        next_qn = A(half_qn, next_pn, h/2)        
        --- calc kinetic energy, calc loss: next_qn -> qn, next_pn -> pn, next_gn -> gn
        half_qn = A(qn, pn, h/2)

        BAD: weights are half-updated

    BAOAB:
        --- force calc
        next_pn = B(tilde_half_pn, next_gn, h/2)
        --- calc kinetic energy, calc loss: next_qn -> qn, next_pn -> pn, next_gn -> gn
        half_pn = B(pn, gn, h/2)
        half_qn = A(qn, half_pn, h/2)
        tilde_half_pn = O(half_pn, rn, alpha1, zeta2)
        next_qn = A(half_qn, tilde_half_pn, h/2)

        BAD: momenta are half-updated
        
    OABAO:
        --- force calc
        tilde_half_pn = B(half_pn, half_gn, h)
        next_qn = A(half_qn, tilde_half_pn, h/2)
        next_pn = O(tilde_half_pn, rn2, alpha2, zeta1)
        --- calc kinetic energy, calc loss: next_qn -> qn, next_pn -> pn, next_gn -> gn
        half_pn = O(pn, rn1, alpha2, zeta1)
        half_qn = A(qn, half_pn, h/2)

        BAD: weights are half-updated
        BAD: momenta are half-updated
        
    OBABO:
        --- force calc
        tilde_next_pn = B(tilde_half_pn, next_gn, h/2)
        next_pn = O(tilde_next_pn, rn2, alpha2, zeta1)
        --- calc kinetic energy, calc loss: next_qn -> qn, next_pn -> pn, next_gn -> gn
        half_pn = O(pn, rn1, alpha2, zeta1)
        tilde_half_pn = B(half_pn, gn, h/2)
        next_qn = A(qn, tilde_half_pn, h)
    
        BAD: momenta are half-updated
           
    We see that we need additional slots to store the half-updated
    parameters.
    Moreover, we need additional nodes that copy the information
    to the parameters. These are called in an extra step, after the 
    train/sample_step and after the loss has been evaluated.
    The kinetic energy is calculated in the train_step itself. Hence,
    it only has to be done at the right moment.

    Hence, before "--- calc kinetic energy, calc loss" I need to
    add all update of parameters fully. After that, the updates always 
    go to the slots update_parameter.
    At the end, loss is evaluated, then I call the node that will place
    update_parameters into parameters and we may continue a new step.
    
    How to do the update_parameters over all layers and weights/bias each?
    '''

    def _apply_dense(self, grads_and_vars, var):
        """Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling using BAOAB.

        BAOAB:

            --- force calc
            next_pn = B(tilde_half_pn, next_gn, h/2)
            --- calc kinetic energy, calc loss: next_qn -> qn, next_pn -> pn, next_gn -> gn
            half_pn = B(pn, gn, h/2)
            half_qn = A(qn, half_pn, h/2)
            tilde_half_pn = O(half_pn, rn, alpha1, zeta2)
            next_qn = A(half_qn, tilde_half_pn, h/2)

        Args:
          grads_and_vars: gradient nodes over all walkers and all variables
          var: parameters of the neural network

        Returns:
          a group of operations to be added to the graph

        """
        precondition_matrix, grad = self._pick_grad(grads_and_vars, var)
        friction_constant_t = math_ops.cast(self._friction_constant_t, var.dtype.base_dtype)
        step_width_t, inverse_temperature_t, random_noise_t = self._prepare_dense(grad, var)

        # tilde_half_pn = p^{n}
        momentum = self.get_slot(var, "momentum")

        scaled_gradient = \
            step_width_t * grad
            # , [grad, var], "precond_grad, var:")

        # next_pn = B(tilde_half_pn, next_gn, h / 2)
        momentum_half_step_t = momentum - 0.5 * scaled_gradient

        # --- calc kinetic energy: next_qn -> qn, next_pn -> pn, next_gn -> gn
        # 1/2 * p^{n}^t * p^{n}
        gradient_global_t = self._get_accumulate_conditional("gradients",
            lambda: self._accumulate_norm("gradients", scaled_gradient))
        virial_global_t = self._get_accumulate_conditional("virials",
            lambda: self._accumulate_scalar_product("virials", grad, var))

        # 1/2 * p^{n}^t * p^{n}
        momentum_sq = tf.reduce_sum(tf.multiply(momentum_half_step_t, momentum_half_step_t))
        kinetic_energy_t = self._get_accumulate_conditional("kinetic_energy",
            lambda: self._accumulate_value("kinetic_energy", momentum_sq, 0.5))
        momentum_global_t = self._get_accumulate_conditional("momenta",
            lambda: self._accumulate_value("momenta", momentum_sq))

        inertia_global_t = self._get_accumulate_conditional("inertia",
            lambda: self._accumulate_scalar_product("inertia", momentum_half_step_t, var))

        # half_pn = B(pn, gn, h/2)
        momentum_full_step_t = \
            momentum_half_step_t - 0.5 * scaled_gradient
            #,[var.name, momentum_half_step_t], "B2: ")
        if len(grads_and_vars) != 1:
            preconditioned_momentum_full_step_t = \
                tf.reshape(
                tf.matmul(precondition_matrix, tf.expand_dims(tf.reshape(momentum_full_step_t, [-1]), 1)),
                var.shape)
                # ,[precondition_matrix], "B2 precond: ")
        else:
            preconditioned_momentum_full_step_t = momentum_full_step_t
        # half_qn = A(qn, half_pn, h / 2)
        position_half_step_t = \
            var + 0.5 * step_width_t * preconditioned_momentum_full_step_t
            #, [var.name, momentum_full_step_t], "B1: ")

        #tilde_half_pn = O(half_pn, rn, alpha1, zeta2)
        alpha_t = tf.exp(-friction_constant_t * step_width_t)
        scaled_noise = tf.sqrt((1.-tf.pow(alpha_t, 2))/inverse_temperature_t) * random_noise_t
        rescaled_noise = scaled_noise/alpha_t

        # conditionally calculate norm of noise
        noise_global_t = self._get_accumulate_conditional("noise",
            lambda: self._accumulate_norm("noise", rescaled_noise))

        momentum_noise_step_t = alpha_t * momentum_full_step_t + scaled_noise
        if len(grads_and_vars) != 1:
            preconditioned_momentum_noise_step_t = tf.reshape(
                tf.matmul(precondition_matrix, tf.expand_dims(tf.reshape(momentum_noise_step_t, [-1]), 1)),
                var.shape)
        else:
            preconditioned_momentum_noise_step_t = momentum_noise_step_t

        # next_qn = A(half_qn, tilde_half_pn, h / 2)
        position_full_step_t = \
            position_half_step_t + 0.5 * step_width_t * preconditioned_momentum_noise_step_t
            #, [var.name, momentum_noise_step_t, position_half_step_t], "O+A1: ")

        # prior force act directly on var
        ub_repell, lb_repell = self._apply_prior(var)
        prior_force = step_width_t * (ub_repell + lb_repell)

        # make sure virial is evaluated before we update variables
        with tf.control_dependencies([virial_global_t, inertia_global_t]):
            # assign parameters
            var_update = \
                state_ops.assign(var, position_full_step_t - prior_force)
                #, [var.name, position_full_step_t], "A2: ")

        # assign moment to slot
        with tf.control_dependencies([kinetic_energy_t]):
            momentum_t = momentum.assign(momentum_noise_step_t)

        # note: these are evaluated in any order, use control_dependencies if required
        return control_flow_ops.group(*[gradient_global_t, inertia_global_t, virial_global_t, kinetic_energy_t,
                                        noise_global_t, momentum_global_t,
                                        var_update, momentum_t])
