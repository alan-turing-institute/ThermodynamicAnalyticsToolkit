from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
import tensorflow as tf

# importing numpy legendre to be able to do Gauss Quadraturei
import numpy as np
import numpy.polynomial.legendre as legendre

from TATi.models.basetype import dds_basetype
from TATi.samplers.geometriclangevinalgorithmfirstordersampler import GeometricLangevinAlgorithmFirstOrderSampler

class InfiniteSwitchSimulatedTemperingSampler(GeometricLangevinAlgorithmFirstOrderSampler):
    """ Implements an Accalerated Geometric Langevin Algorithm Momentum Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    """
    def __init__(self, step_width, inverse_temperature, friction_constant,
                 inverse_temperature_max,
                 number_of_temperatures, alpha_constant,
                 seed=None, use_locking=False, name='InfiniteSwitchSimulatedTempering'):
        """ Init function for this class.

        :param step_width: step width for gradient, also affects inject noise
        :param inverse_temperature: scale for gradients, also lower integration boundary
        :param friction_constant: scales the momenta
        :param inverse_temperature_max: upper integration boundary
        :param number_of_temperatures: number of interpolation points for temperature integration
        :param alpha_constant: constant scaling the weight-optimisation timestep
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(InfiniteSwitchSimulatedTemperingSampler, self).__init__(step_width,
                                                                      inverse_temperature, friction_constant,
                                                                      seed, use_locking, name)
        self._inverse_temperature_max = inverse_temperature_max
        self._number_of_temperatures = number_of_temperatures
        self._alpha_constant = alpha_constant

        # get the weights and the sample points for
        # Gauss-Legendre integration on [-1,1]
        points,weights = legendre.leggauss(number_of_temperatures)

        # scale the points and the weights into performing
        # integration on the correct region [a,b]
        a = inverse_temperature
        b = inverse_temperature_max

        Beta = np.add(np.multiply(0.5*(b-a), points), 0.5*(a+b))
        Gweight = np.multiply(0.5*(b-a),weights)

        # convert the numpy arrays into tensors
        self._isst_beta = ops.convert_to_tensor(Beta, dtype=dds_basetype, name="isst_beta")
        self._gauss_weight = ops.convert_to_tensor(Gweight, dtype=dds_basetype, name="gauss_weight")
        
        # intialise the isst average
        self._isst_average = tf.get_variable(shape=[number_of_temperatures],
                                             dtype=dds_basetype,
                                             name='isst_average',
                                             initializer=tf.zeros_initializer)        

        # initialise the isst weights
        self._isst_weight = tf.get_variable(shape=[number_of_temperatures],
                                            dtype=dds_basetype,
                                            name='isst_weight',
                                            initializer=tf.ones_initializer)

        # calculate scaling and reshape to scalar
        WeightMul = tf.reduce_sum(tf.multiply(self._gauss_weight,self._isst_weight),0)

        # resecale the weights
        tf.assign(self._isst_weight,tf.scalar_mul(1.0/WeightMul,self._isst_weight))
        
        # convert alpha into a tensor if given as a float 
        self._alpha_constant = ops.convert_to_tensor(self._alpha_constant, name="alpha_constant")

        # initialise the step index used to calculate the mean
        self._step_index = 0
        
    def _get_force_rescaling(self,loss):
        """ Updates the force rescaling needed to rescale the force with 
        to perform the integration.
        """
        expV = tf.exp(tf.scalar_mul(loss,self._isst_beta))

        BarNumSum = tf.reduce_sum(tf.multiply(tf.multiply(self._gauss_weight,self._isst_beta),
                                              tf.multiply(self._isst_weight,expV)))
        BarDeNumSum = tf.reduce_sum(tf.multiply(self._gauss_weight,
                                                tf.multiply(self._isst_weight,expV)))
        thermal_scaling = 0.0
        
        tf.assign(thermal_scaling, BarNumSum / (self._inverse_temperature * BarDeNumSum))

        return thermal_scaling
        
    def _update_learn_weights(self,loss):
        """ Updates the isst_weights stored in the InfiniteSwitchSimulatedTemperingSampler
        class. 
        """
        # update the step index counter
        self._step_index = self._step_index + 1 

        expV = tf.exp(tf.scalar_mul(loss, self._isst_beta))
        BarDeNumSum = tf.reduce_sum(tf.multiply(self._gauss_weight,
                                                tf.multiply(self._isst_weight, expV)))

        # add the the running average
        tf.assign_add(self._isst_average,tf.scalar_mul(1.0/BarDeNumSum,expV))

        # calculate the inverse weights
        tf.assign(self._isst_weight,tf.reciprocal(tf.add(tf.scalar_mul(1.0-self._alpha_constant * self._step_width,
                                                                       tf.reciprocal(self._isst_weight)),
                                                         tf.scalar_mul(1.0/self._step_index, self._isst_average))))
        
        # calculate scaling
        WeightMul = tf.reduce_sum(tf.multiply(self._gauss_weight,self._isst_weight),0)
        
        # resecale the weights
        tf.assign(self._isst_weight,tf.scalar_mul(1.0/WeightMul,self._isst_weight))
        
    def _apply_dense(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling using BAOAB.

        BAOAB:
            --- force calc
            next_pn = B(tilde_half_pn, next_qn, h/2)
            --- calc kinetic energy, calc loss: next_qn -> qn, next_pn -> pn, next_gn -> gn
            half_pn = B(pn, gn, h/2)
            half_qn = A(qn, half_pn, h/2)
            tilde_half_pn = O(half_pn, rn, alpha1, zeta2)
            next_qn = A(half_qn, tilde_half_pn, h/2)

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        friction_constant_t = math_ops.cast(self._friction_constant_t, var.dtype.base_dtype)
        step_width_t, inverse_temperature_t, random_noise_t = self._prepare_dense(grad, var)

        # tilde_half_pn = p^{n}
        momentum = self.get_slot(var, "momentum")

        # senare - this might not be the right place nor order
        #
        ##_____ ISST _____##

        # get the loss scalar
        with tf.variable_scope("accumulate", reuse=True):
            loss = tf.get_variable("old_loss", dtype=dds_basetype)
            
        # updates the weight learning held by the class
        self._update_learn_weights(loss)

        # get's the isst_scaling which is a scalar
        isst_scaling = self._get_force_rescaling(loss)

        ##_____ ISST _____##
        
        scaled_gradient = step_width_t * isst_scaling * grad

        # next_pn = B(tilde_half_pn, next_gn, h / 2)
        momentum_half_step_t = momentum - 0.5 * scaled_gradient

        # --- calc kinetic energy: next_qn -> qn, next_pn -> pn, next_gn -> gn

        # half_pn = B(pn, gn, h/2)
        momentum_full_step_t =  momentum_half_step_t - 0.5 * scaled_gradient

        # half_qn = A(qn, half_pn, h / 2)
        position_half_step_t = var + 0.5 * step_width_t * momentum_full_step_t

        #tilde_half_pn = O(half_pn, rn, alpha1, zeta2)
        alpha_t = tf.exp(-friction_constant_t * step_width_t)
        scaled_noise = tf.sqrt((1.-tf.pow(alpha_t, 2))/inverse_temperature_t) * random_noise_t

        with tf.variable_scope("accumulate", reuse=True):
            noise_global = tf.get_variable("noise", dtype=dds_basetype)
            noise_global_t = tf.assign_add(noise_global, tf.pow(alpha_t, -2) *
                                           tf.reduce_sum(tf.multiply(scaled_noise, scaled_noise)))
        
        momentum_noise_step_t = alpha_t * momentum_full_step_t + scaled_noise

        # next_qn = A(half_qn, tilde_half_pn, h / 2)
        position_full_step_t = position_half_step_t + 0.5 * step_width_t * momentum_noise_step_t

        # prior force act directly on var
        ub_repell, lb_repell = self._apply_prior(var)
        prior_force = step_width_t * (ub_repell + lb_repell)

        # make sure virial and gradients are evaluated before we update variables
        #with tf.control_dependencies([virial_global_t, gradient_global_t]):
        # assign parameters
        var_update = state_ops.assign(var, position_full_step_t - prior_force)

        # assign moment to slot
        #with tf.control_dependencies([kinetic_energy_t]):
        momentum_t = momentum.assign(momentum_noise_step_t)

        # note: these are evaluated in any order, use control_dependencies if required
        return control_flow_ops.group(*[var_update, momentum_t])

    def _apply_sparse(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of sparsely
        occupied tensors to perform the actual sampling.
        
        Note that this is not implemented so far.
        
        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        raise NotImplementedError("Sparse gradient updates are not supported.")
