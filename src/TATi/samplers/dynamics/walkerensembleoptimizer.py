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

from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.training.optimizer import Optimizer, _DenseResourceVariableProcessor, \
    _OptimizableVariable
import tensorflow as tf

from distutils.version import LooseVersion

if LooseVersion(tf.__version__) >= LooseVersion("1.7.0"):
    from tensorflow.python.training.optimizer import _TensorProcessor

if LooseVersion(tf.__version__) < LooseVersion("1.8.0"):
    from tensorflow.python.training.optimizer import _get_variable_for

if LooseVersion(tf.__version__) < LooseVersion("1.9.0"):
    from tensorflow.python.training.optimizer import _StreamingModelPortProcessor

import collections

from TATi.models.basetype import dds_basetype

class RefVariableWalkerEnsembleProcessor(_OptimizableVariable):
    """Processor for List of WalkerEnsemble Variable.

    This variable has multiple related gradients coming from the replicated
    versions of the neural network, each one associated to a walker in an
    ensemble of walkers, which might be used in the update scheme. This class
    overrides the default `update_op` of `tensorflow`'s `Optimzer` to allow
    for list of tensor to be passed down.
    """

    def __init__(self, v):
        self._v = v

    def __str__(self):
        return "<_RefVariableProcessor(%s)>" % self._v

    def target(self):
        return self._v._ref()  # pylint: disable=protected-access

    def update_op(self, optimizer, grads_and_vars):
        if not isinstance(grads_and_vars, ops.IndexedSlices):
            update_op = optimizer._apply_dense(grads_and_vars, self._v)  # pylint: disable=protected-access
            if self._v.constraint is not None:
                with ops.control_dependencies([update_op]):
                    return self._v.assign(self._v.constraint(self._v))
            else:
                return update_op
        else:
            if self._v.constraint is not None:
                raise RuntimeError(
                    "Cannot use a constraint function on a sparse variable.")
            # pylint: disable=protected-access
            return optimizer._apply_sparse_duplicate_indices(grads_and_vars, self._v)

def _get_processor(v):
    """The processor of v."""
    if LooseVersion(tf.__version__) >= LooseVersion("1.7.0"):
        if context.executing_eagerly():
            if isinstance(v, ops.Tensor):
                return _TensorProcessor(v)
            else:
                return _DenseResourceVariableProcessor(v)
    else:
        if context.in_eager_mode():
            return _DenseResourceVariableProcessor(v)
    if v.op.type == "VarHandleOp":
        return _DenseResourceVariableProcessor(v)
    if isinstance(v, variables.Variable):
        return RefVariableWalkerEnsembleProcessor(v)
    if LooseVersion(tf.__version__) < LooseVersion("1.9.0"):
        if v.op.type == "SubmodelPort":
            return _StreamingModelPortProcessor(v)
    if LooseVersion(tf.__version__) >= LooseVersion("1.7.0"):
        if isinstance(v, ops.Tensor):
            return _TensorProcessor(v)
    raise NotImplementedError("Trying to optimize unsupported type ", v)

class WalkerEnsembleOptimizer(Optimizer):
    """ This implements a version of the optimizer that has more than one walkers
    that exchange their gradients among one another.

    We override the `Optimizer` class in `tensorflow` in order to wedge in
    functionality between computation of the gradients (i.e. adding nodes to
    the computational graph that do this) and the actual application of the
    gradients for doing the position update.

    This way the position update may actually rely on the gradients of other
    instances (or any other information) stored in possible walkers.
    """
    def __init__(self, covariance_blending, use_locking=False, name='WalkerEnsembleOptimizer'):
        """ Init function for this class.

        :param covariance_blending: covariance identity blending value eta to use in creating the preconditioning matrix
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(WalkerEnsembleOptimizer, self).__init__(use_locking, name)
        self._covariance_blending = covariance_blending
        self.EQN_update = None

    def _prepare(self):
        """ Converts step width into a tensor, if given as a floating-point
        number.
        """
        super(WalkerEnsembleOptimizer, self)._prepare()
        self._covariance_blending_t = ops.convert_to_tensor(self._covariance_blending, name="covariance_blending")

    def compute_and_check_gradients(self, loss, var_list=None,
                 gate_gradients=Optimizer.GATE_GRAPH, aggregation_method=None,
                 colocate_gradients_with_ops=False, grad_loss=None):
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops" \
                +" that do not support gradients, between variables %s and loss %s." \
                % ([str(v) for _, v in grads_and_vars], loss))

        return grads_and_vars

    def apply_gradients(self, walkers_grads_and_vars, index, global_step=None, name=None):
        """Apply gradients to variables.

        Here, we actually more gradient nodes than we require, namely addtionally
        those from all other walkers. We pick the ones associated with each
        specific variable, e.g. all gradients to each layer1 weights in all
        walkers. And these are given to the internally called `apply_dense`
        function.

        :param grads_and_vars: gradients and variables as a list over all parallel
                        walkers
        :param index: index of this walker
        """
        # add processor for each variable
        grads_and_vars = tuple(walkers_grads_and_vars[index])    # Make sure repeat iteration works.
        if not grads_and_vars:
            raise ValueError("No variables provided.")
        converted_grads_and_vars = []
        for g, v in grads_and_vars:
            if g is not None:
                try:
                    # Convert the grad to Tensor or IndexedSlices if necessary.
                    g = ops.convert_to_tensor_or_indexed_slices(g)
                except TypeError:
                    raise TypeError(
                            "Gradient must be convertible to a Tensor"
                            " or IndexedSlices, or None: %s" % g)
                if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
                    raise TypeError(
                            "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
            p = _get_processor(v)
            converted_grads_and_vars.append((g, v, p))

        # check for present gradients
        converted_grads_and_vars = tuple(converted_grads_and_vars)
        var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]
        if not var_list:
            raise ValueError("No gradients provided for any variable: %s." %
                                             ([str(v) for _, _, v in converted_grads_and_vars],))

        # create slots for each variable
        if LooseVersion(tf.__version__) >= LooseVersion("1.8.0"):
            with ops.init_scope():
                self._create_slots(var_list)
        elif LooseVersion(tf.__version__) >= LooseVersion("1.7.0"):
            with ops.init_scope():
                self._create_slots([_get_variable_for(v) for v in var_list])
        else:
            with ops.control_dependencies(None):
                self._create_slots([_get_variable_for(v) for v in var_list])

        # create the (position) update operations
        update_ops = []
        with ops.name_scope(name, self._name) as name:
            self._prepare()
            # for each variable, call `processor.update_up()`
            for grad, var, processor in converted_grads_and_vars:
                if grad is None:
                    continue
                # We colocate all ops created in _apply_dense or _apply_sparse
                # on the same device as the variable.
                if LooseVersion(tf.__version__) >= LooseVersion("1.7.0"):
                    scope_name = "" if context.executing_eagerly() else var.op.name
                else:
                    scope_name = var.op.name if context.in_graph_mode() else ""
                with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
                    update_ops.append(processor.update_op(self, walkers_grads_and_vars))

            # append global_step increment
            if global_step is None:
                apply_updates = self._finish(update_ops, name)
            else:
                with ops.control_dependencies([self._finish(update_ops, "update")]):
                    with ops.colocate_with(global_step):
                        if isinstance(global_step, resource_variable_ops.ResourceVariable):
                            # TODO(apassos): the implicit read in assign_add is slow; consider
                            # making it less so.
                            apply_updates = resource_variable_ops.assign_add_variable_op(
                                global_step.handle,
                                ops.convert_to_tensor(1, dtype=global_step.dtype),
                                name=name)
                        else:
                            apply_updates = state_ops.assign_add(global_step, 1, name=name)

            # put into collection and return
            if LooseVersion(tf.__version__) >= LooseVersion("1.7.0") and \
                    not context.executing_eagerly():
                if isinstance(apply_updates, ops.Tensor):
                    apply_updates = apply_updates.op
            else:
                apply_updates = apply_updates.op

            if LooseVersion(tf.__version__) < LooseVersion("1.7.0") or \
                    not context.executing_eagerly():
                train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
                if apply_updates not in train_op:
                    train_op.append(apply_updates)

            return apply_updates

    def _get_preconditioner(self, flat_othervars, var):
        covariance_blending_t = math_ops.cast(self._covariance_blending_t, var.dtype.base_dtype)
        _, cov = self._get_covariance(flat_othervars, var)
        # \sqrt{ 1_D + \eta cov(flat_othervars)}, see [Matthews, Weare, Leimkuhler, 2018]
        unity = tf.matrix_band_part(tf.ones_like(cov, dtype=dds_basetype), 0, 0)
        matrix = unity + covariance_blending_t * cov
        max_matrix = tf.reduce_max(tf.abs(matrix))

        def accept():
            return tf.reciprocal(max_matrix)

        def reject():
            return max_matrix

        normalizing_factor = tf.cond(tf.greater(max_matrix, 1.),
                                     accept, reject)
        update_precond = tf.sqrt(tf.reciprocal(normalizing_factor)) * \
                tf.cholesky(normalizing_factor * matrix)

        scope_name = "EQN_%s" % (var.name[:var.name.find(":")].replace("/", "_"))
        with tf.variable_scope(scope_name, reuse=True):
            precondition_matrix = tf.get_variable("precondition_matrix", dtype=dds_basetype)

        precond_assign = precondition_matrix.assign(update_precond) #, name="update_preccondition_matrix")

        return tf.Print(
            precond_assign, [precond_assign], "preconditioner:", summarize=9)

    def _pick_grad(self, grads_and_vars, var):
        """ Helper function to extract the gradient associated with `var` from
        the list containing all gradients and variables in `grads_and_vars`.

        If `do_precondition_gradient` is true, then this will additionally
        precondition the gradient using the other walkers' gradients.

        :param grads_and_vars: list of grad, var tuples over all walkers
        :param var: current variable to associated to grad of this walker instance
        :return: (preconditioned) grad associated to var
        """
        #print(var.name[var.name.find("/"):])
        grad, _ = self._extract_grads(grads_and_vars, var)
        #print("grad: "+str(grad))
        #print("grads_and_vars: "+str(grads_and_vars))
        _, flat_othervars = self._extract_vars(grads_and_vars, var)
        vars = tf.stack(flat_othervars)
        number_walkers = vars.shape[0]
        #print("picked_var: "+str(picked_var))
        def single_walker():
            return tf.constant(1)
        def multi_walker():
            return tf.size(var)
        #print("flat_othervars: "+str(flat_othervars))
        number_dim = tf.cond(
            tf.equal(0, number_walkers),
            single_walker, multi_walker)
        #print(number_dim)
        precondition_matrix_initial = tf.eye(number_dim) # flat_othervars[0].shape[0])
        scope_name = "EQN_%s" % (var.name[:var.name.find(":")].replace("/", "_"))
        with tf.variable_scope(scope_name, reuse=False):
            precondition_matrix = tf.get_variable(initializer=precondition_matrix_initial,
                                                  validate_shape=False,  # shape is run-tine initialized
                                                  trainable=False, dtype=dds_basetype,
                                                  name="precondition_matrix")

        if len(flat_othervars) != 0:

            # create the preconditioning matrix update node
            self.EQN_update = self._get_preconditioner(flat_othervars, var)

            # apply the covariance matrix to the flattened gradient and then return to
            # original shape to match for variable update

            def precondition_grad():
                return tf.reshape(
                    tf.matmul( precondition_matrix, tf.expand_dims(tf.reshape(grad, [-1]), 1), transpose_a=True),
                    grad.get_shape())
            def pass_grad():
                return tf.identity(grad)

            # matrix-vector multiplication is however a bit more complicated, see
            # https://stackoverflow.com/a/43285258/1967646
            preconditioned_grad = tf.cond(
                tf.equal(0, number_walkers),
                pass_grad, precondition_grad)

            return precondition_matrix, preconditioned_grad
        else:
            return precondition_matrix, grad

    def _extract_grads(self, grads_and_vars, var):
        """ Helper function to extract the gradient associated with `var` from
        the list containing all gradients and variables in `grads_and_vars`.
        Moreover, we return also all other gradients (in walkers) associated to
        the same var.

        :param grads_and_vars: list of grad, var tuples over all walkers
        :param var: current variable
        :return: grad associated to `var`,
                other grads associated to equivalent var in other walkers
        """
        #print("var: "+str(var))
        #print("truncated var: "+str(var.name[var.name.find("/"):]))
        grad = None
        othergrads = []
        for i in range(len(grads_and_vars)):
            for g, v in grads_and_vars[i]:
                if g is not None:
                    if v.name == var.name:
                        grad = g
                    elif v.name[v.name.find("/"):] == var.name[var.name.find("/"):]:
                        othergrads.append(g)
        return grad, othergrads

    def _extract_vars(self, grads_and_vars, var):
        """ Helper function to extract the variable associated with `var` from
        the list containing all gradients and variables in `grads_and_vars`.
        Moreover, we return also all other gradients (in walkers) associated to
        the same var.

        :param grads_and_vars: list of grad, var tuples over all walkers
        :param var: current variable
        :return: other vars associated to equivalent var in other walkers
        """
        othervars = []
        for i in range(len(grads_and_vars)):
            for g, v in grads_and_vars[i]:
                if v is not None:
                    if v.name == var.name:
                        pass
                    elif v.name[v.name.find("/"):] == var.name[var.name.find("/"):]:
                        #print("Appending to othervars: "+str(v))
                        othervars.append(tf.reshape(v, [-1], name=v.name[:v.name.find(":")]+"/reshape"))
        return var, othervars

    def _get_covariance(self, flat_othervars, var):
        """ Returns node for the covariance between the variables of all other
        walkers plus the identity.

        :param flat_othervars: list of all other variables
        :return: node for the stacked variables and the covariance matrix
        """
        # we use a flat vars tensor as otherwise the index arithmetics becomes
        # very complicated for the "matrix * vector" product (covariance matrix
        # times the gradient) in the end. This is undone in `_pick_grad()`
        #print(flat_othervars)
        vars = tf.stack(flat_othervars)
            # tf.Print(,
            # [flat_othervars], "flat_othervars", summarize=10)

        number_dim = tf.size(flat_othervars[0])
        number_walkers = vars.shape[0]
        norm_factor = self._get_norm_factor(number_dim, number_walkers)

        means = self._get_means(vars)
        rank1factors = self._get_rank1_factors(vars, means)
        cov = tf.matmul(
            tf.transpose(rank1factors),
            rank1factors,
            name="rank1factor_product"
        )
        #return vars, norm_factor * tf.Print(cov, [cov.name, cov], "cov: ", summarize=9)
        return vars, norm_factor * cov

    def _get_norm_factor(self, number_dim, number_walkers):
        dim = math_ops.cast(number_walkers
                            # tf.Print(number_walkers, [number_walkers], "number_walkers: ")
                            , dtype=dds_basetype)

        def accept_block():
            return tf.reciprocal(dim)

        def reject_block():
            return tf.constant(1.)

        norm_factor = tf.cond(
            tf.greater(number_dim, 1),
            accept_block, reject_block)
            # tf.Print(,[number_dim], "number_dim: ")

        return norm_factor

    def _get_means(self, vars):
        # means are needed for every component. Hence, we save a bit by
        # creating all means beforehand
        means = tf.reduce_mean(vars, axis=0, name="reduce_mean_over_othervars")

        #return tf.Print(means, [means], "mean: ", summarize=4)
        return means

    def _get_rank1_factors(self, vars, means):
        stacked_means = tf.tensordot(
            tf.expand_dims(tf.ones_like(means, name="ones_for_broadcast"), 0),
            tf.expand_dims(means, 0),
            axes=[[0], [0]], name="broadcast_means")
        rank1factors = vars - stacked_means
        #return tf.Print(rank1factors, [rank1factors], "assigned factors", summarize=4)
        return rank1factors

    def _stack_gradients(self, grads_and_vars, var):
        """ Stacks all (other) gradients together to compute a covariance matrix.

        :param grads_and_vars: list of gradients and vars from all walkers
        :param var: this walker's variable
        :return: stacked gradients from _other_ walkers (excluding the one
                associated to var)
        """
        grads = []
        for i in range(len(grads_and_vars)):
            for g, v in grads_and_vars[i]:
                if g is not None:
                    if v.name == var.name:
                        grads.append(tf.zeros(var.shape, dtype=var.dtype))
                    elif v.name[v.name.find("/"):] == var.name[var.name.find("/"):]:
                        grads.append(g)
        #print(grads)
        return tf.stack(grads)

    @staticmethod
    def get_covariance_component(x,y):
        """ Calculates the covariance between two variables `x` and `y`.
-
        :param x: first variable
        :param y: second variable
        :return: cov(x,y)
        """
        dim = math_ops.cast(tf.size(x), dds_basetype)
        def accept_block():
            return tf.reciprocal(dim - 1.)

        def reject_block():
            return tf.constant(1.)

        norm_factor = tf.cond(
            tf.greater(tf.size(x), 1),
            accept_block, reject_block)

        return norm_factor * \
               tf.reduce_sum((x - tf.reduce_mean(x)) * (y - tf.reduce_mean(y)))
