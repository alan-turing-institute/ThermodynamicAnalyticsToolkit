from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
import tensorflow as tf

from DataDrivenSampler.samplers.GLAFirstOrderMomentumSampler import GLAFirstOrderMomentumSampler


class BAOABSampler(GLAFirstOrderMomentumSampler):
    """ Implements a Geometric Langevin Algorithm Momentum Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    """
    def __init__(self, step_width, inverse_temperature, friction_constant,
                 seed=None, use_locking=False, name='BAOAB'):
        """ Init function for this class.

        :param step_width: step width for gradient, also affects inject noise
        :param inverse_temperature: scale for gradients
        :param friction_constant: scales the momenta
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(BAOABSampler, self).__init__(step_width, inverse_temperature,
                                                            friction_constant, seed, use_locking, name)


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

    def _apply_dense(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling using BAOAB.

        BAOAB:
            --- force calc
            next_pn = B(tilde_half_pn, next_gn, h/2)
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

        scaled_gradient = step_width_t * grad

        # next_pn = B(tilde_half_pn, next_gn, h / 2)
        momentum_half_step_t = momentum - 0.5 * scaled_gradient

        # --- calc kinetic energy: next_qn -> qn, next_pn -> pn, next_gn -> gn
        # 1/2 * p^{n}^t * p^{n}
        momentum_sq = 0.5 * tf.reduce_sum(tf.multiply(momentum_half_step_t, momentum_half_step_t))
        with tf.variable_scope("accumulate", reuse=True):
            kinetic_energy = tf.get_variable("kinetic", dtype=tf.float64)
            kinetic_energy_t = tf.assign_add(kinetic_energy, momentum_sq)

        with tf.variable_scope("accumulate", reuse=True):
            gradient_global = tf.get_variable("gradients", dtype=tf.float64)
            gradient_global_t = tf.assign_add(gradient_global, tf.reduce_sum(tf.multiply(scaled_gradient, scaled_gradient)))
            # configurational temperature
            virial_global = tf.get_variable("virials", dtype=tf.float64)
            virial_global_t = tf.assign_add(virial_global, tf.reduce_sum(tf.multiply(grad, var)))

        with tf.variable_scope("accumulate", reuse=True):
            momentum_global = tf.get_variable("momenta", dtype=tf.float64)
            momentum_global_t = tf.assign_add(momentum_global, tf.reduce_sum(tf.multiply(momentum_half_step_t, momentum_half_step_t)))

        # half_pn = B(pn, gn, h/2)
        momentum_full_step_t =  momentum_half_step_t - 0.5 * scaled_gradient

        # half_qn = A(qn, half_pn, h / 2)
        position_half_step_t = var + 0.5 * step_width_t * momentum_full_step_t

        #tilde_half_pn = O(half_pn, rn, alpha1, zeta2)
        alpha_t = tf.exp(-friction_constant_t * step_width_t)
        scaled_noise = tf.sqrt((1.-tf.pow(alpha_t, 2))/inverse_temperature_t) * random_noise_t
        with tf.variable_scope("accumulate", reuse=True):
            noise_global = tf.get_variable("noise", dtype=tf.float64)
            noise_global_t = tf.assign_add(noise_global, tf.pow(alpha_t, -2)*tf.reduce_sum(tf.multiply(scaled_noise, scaled_noise)))
        momentum_noise_step_t = alpha_t * momentum_full_step_t + scaled_noise

        # next_qn = A(half_qn, tilde_half_pn, h / 2)
        position_full_step_t = position_half_step_t + 0.5 * step_width_t * momentum_noise_step_t

        # prior force act directly on var
        ub_repell, lb_repell = self._apply_prior(var)
        prior_force = step_width_t * (ub_repell + lb_repell)

        # make sure virial and gradients are evaluated before we update variables
        with tf.control_dependencies([virial_global_t, gradient_global_t, kinetic_energy_t]):
            # assign parameters
            var_update = state_ops.assign(var, position_full_step_t - prior_force)

        # assign moment to slot
        momentum_t = momentum.assign(momentum_noise_step_t)

        # note: these are evaluated in any order, use control_dependencies if required
        return control_flow_ops.group(*[gradient_global_t, virial_global_t,
                                        noise_global_t, momentum_global_t,
                                        kinetic_energy_t,
                                        var_update, momentum_t])
