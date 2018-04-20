# This is heavily inspired by  https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
import tensorflow as tf

from TATi.models.basetype import dds_basetype
from TATi.samplers.replicaoptimizer import ReplicaOptimizer


class StochasticGradientLangevinDynamicsSampler(ReplicaOptimizer):
    """ Implements a Stochastic Gradient Langevin Dynamics Sampler
    in the form of a TensorFlow Optimizer, overriding tensorflow.python.training.Optimizer.

    We have the upgrade as: %\Delta\Theta = - \nabla \beta U(\Theta) \Delta t + \sqrt{\Delta t} N$,
    where $\beta$ is the inverse temperature coefficient, $\Delta t$ is the (discretization)
    step width and $\Theta$ is the parameter vector and $U(\Theta)$ the energy or loss function.
    """
    def __init__(self, ensemble_precondition, step_width, inverse_temperature,
                 seed=None, use_locking=False, name='SGLD'):
        """ Init function for this class.

        :param ensemble_precondition: whether to precondition the gradient using
                all the other replica or not
        :param step_width: step width for gradient, also affects inject noise
        :param inverse_temperature: scale for gradients
        :param seed: seed value of the random number generator for generating reproducible runs
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(StochasticGradientLangevinDynamicsSampler, self).__init__(ensemble_precondition,
                                                                        use_locking, name)
        self._step_width = step_width
        self._seed = seed
        self.random_noise = None
        self.scaled_gradient = None
        self.scaled_noise = None
        self._inverse_temperature = inverse_temperature
        self.upper_boundary = None
        self.lower_boundary = None
        self.force_factor = .1
        self.force_power = 1.

    def _prepare(self):
        """ Converts step width into a tensor, if given as a floating-point
        number.
        """
        self._step_width_t = ops.convert_to_tensor(self._step_width, name="step_width")
        self._inverse_temperature_t = ops.convert_to_tensor(self._inverse_temperature, name="inverse_temperature")

    def _create_slots(self, var_list):
        """ Slots are internal resources for the Optimizer to store values
        that are required and modified during each iteration.

        Here, we do not create any slots.

        :param var_list: list of variables
        """
        pass

    def _prepare_dense(self, grad, var):
        """ Stuff common to all Langevin samplers.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: step_width, inverse_temperature, and noise tensors
        """
        step_width_t = math_ops.cast(self._step_width_t, var.dtype.base_dtype)
        inverse_temperature_t = math_ops.cast(self._inverse_temperature_t, var.dtype.base_dtype)
        if self._seed is None:
            random_noise_t = tf.random_normal(grad.get_shape(), mean=0.,stddev=1., dtype=dds_basetype)
        else:
            # increment such that we use different seed for each random tensor
            self._seed += 1
            random_noise_t = tf.random_normal(grad.get_shape(), mean=0., stddev=1., dtype=dds_basetype, seed=self._seed)
        return step_width_t, inverse_temperature_t, random_noise_t

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

    def _apply_dense(self, grads_and_vars, var):
        """ Adds nodes to TensorFlow's computational graph in the case of densely
        occupied tensors to perform the actual sampling.

        We simply add nodes that generate normally distributed noise here, one
        for each weight.
        The norm of the injected noise is placed into the TensorFlow summary.

        :param grads_and_vars: gradient nodes over all replicas and all variables
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        # Pick correct gradient from grad_list
        #print(grad)
        #print(othergrads)
        grad = self._pick_grad(grads_and_vars, var)
        step_width_t, inverse_temperature_t, random_noise_t = self._prepare_dense(grad, var)
        # \nabla V (q^n ) \Delta t
        scaled_gradient = step_width_t * grad

        with tf.variable_scope("accumulate", reuse=True):
            gradient_global = tf.get_variable("gradients", dtype=dds_basetype)
            gradient_global_t = tf.assign_add(gradient_global, tf.reduce_sum(tf.multiply(scaled_gradient, scaled_gradient)))
            # configurational temperature
            virial_global = tf.get_variable("virials", dtype=dds_basetype)
            virial_global_t = tf.assign_add(virial_global, tf.reduce_sum(tf.multiply(grad, var)))

        scaled_noise = tf.sqrt(2.*step_width_t/inverse_temperature_t) * random_noise_t
        with tf.variable_scope("accumulate", reuse=True):
            noise_global = tf.get_variable("noise", dtype=dds_basetype)
            noise_global_t = tf.assign_add(noise_global, tf.reduce_sum(tf.multiply(scaled_noise, scaled_noise)))

        # prior force act directly on var
        ub_repell, lb_repell = self._apply_prior(var)
        prior_force = step_width_t * (ub_repell + lb_repell)

        # make sure virial is evaluated before we update variables
        with tf.control_dependencies([virial_global_t]):
            var_update = state_ops.assign_sub(var, scaled_gradient + scaled_noise + prior_force)

        # note: these are evaluated in any order, use control_dependencies if required
        return control_flow_ops.group(*[virial_global_t, gradient_global_t, noise_global_t,
                                        var_update])

    def _apply_sparse(self, grad, var):
        """ Adds nodes to TensorFlow's computational graph in the case of sparsely
        occupied tensors to perform the actual sampling.

        Note that this is not implemented so far.

        :param grad: gradient nodes, i.e. they contain the gradient per parameter in `var`
        :param var: parameters of the neural network
        :return: a group of operations to be added to the graph
        """
        raise NotImplementedError("Sparse gradient updates are not supported.")
