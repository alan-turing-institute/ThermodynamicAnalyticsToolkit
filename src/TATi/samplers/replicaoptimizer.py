from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.training.optimizer import Optimizer, _DenseResourceVariableProcessor, \
    _OptimizableVariable, _StreamingModelPortProcessor, _get_variable_for
import tensorflow as tf

from TATi.models.basetype import dds_basetype

class RefVariableReplicaProcessor(_OptimizableVariable):
    """Processor for List of replicated Variable.

    This variable has multiple related gradients coming from the replicated
    versions which might be used in the update scheme. This class overrides
    the default `update_op` of `tensorflow`'s `Optimzer` to allow for list
    of tensor to be passed down.
    """

    def __init__(self, v):
        self._v = v

    def target(self):
        return self._v._ref()  # pylint: disable=protected-access

    def update_op(self, optimizer, grads_and_vars):
        update_op = optimizer._apply_dense(grads_and_vars, self._v)  # pylint: disable=protected-access
        if self._v.constraint is not None:
            with ops.control_dependencies([update_op]):
                return self._v.assign(self._v.constraint(self._v))
        else:
            return update_op

def _get_processor(v):
    """The processor of v."""
    if context.in_eager_mode():
        return _DenseResourceVariableProcessor(v)
    if v.op.type == "VarHandleOp":
        return _DenseResourceVariableProcessor(v)
    if isinstance(v, variables.Variable):
        return RefVariableReplicaProcessor(v)
    if v.op.type == "SubmodelPort":
        return _StreamingModelPortProcessor(v)
    raise NotImplementedError("Trying to optimize unsupported type ", v)

class ReplicaOptimizer(Optimizer):
    """ This implements a version of the optimizer that has more than one replica
    that exchange their gradients among one another.

    We override the `Optimizer` class in `tensorflow` in order to wedge in
    functionality between computation of the gradients (i.e. adding nodes to
    the computational graph that do this) and the actual application of the
    gradients for doing the position update.

    This way the position update may actually rely on the gradients of other
    instances (or any other information) stored in possible replica.
    """
    def __init__(self, ensemble_precondition=False, use_locking=False, name='ReplicaOptimizer'):
        """ Init function for this class.

        :param ensemble_precondition: whether to precondition the gradient using
                all the other replica or not
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(ReplicaOptimizer, self).__init__(use_locking, name)
        self.ensemble_precondition = ensemble_precondition

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

    def apply_gradients(self, replicated_grads_and_vars, index, global_step=None, name=None):
        """Apply gradients to variables.

        Here, we actually more gradient nodes than we require, namely addtionally
        those from all other replica. We pick the ones associated with each
        specific variable, e.g. all gradients to each layer1 weights in all
        replica. And these are given to the internally called `apply_dense`
        function.

        :param grads_and_vars: gradients and variables as a list over all parallel
                        replica
        :param global_step: global_step node for this replica
        :param name: name of this operation
        """
        # add processor for each variable
        grads_and_vars = tuple(replicated_grads_and_vars[index])    # Make sure repeat iteration works.
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
                scope_name = var.op.name if context.in_graph_mode() else ""
                with ops.name_scope("update_" + scope_name), ops.colocate_with(var):
                    update_ops.append(processor.update_op(self, replicated_grads_and_vars))

            # append global_step increment
            if global_step is None:
                apply_updates = self._finish(update_ops, name)
            else:
                with ops.control_dependencies([self._finish(update_ops, "update")]):
                    with ops.colocate_with(global_step):
                        apply_updates = state_ops.assign_add(global_step, 1, name=name).op

            # put into collection and return
            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            if apply_updates not in train_op:
                train_op.append(apply_updates)

            return apply_updates

    def _pick_grad(self, grads_and_vars, var):
        """ Helper function to extract the gradient associated with `var` from
        the list containing all gradients and variables in `grads_and_vars`.

        If `do_precondition_gradient` is true, then this will additionally
        precondition the gradient using the other replica's gradients.

        :param grads_and_vars: list of grad, var tuples over all replica
        :param var: current variable to associated to grad of this replica instance
        :return: (preconditioned) grad associated to var
        """
        #print(var.name[var.name.find("/"):])
        eta = 1.
        grad, othergrads = self._extract_grads(grads_and_vars, var)
        preconditioned_grad = grad
        if self.ensemble_precondition:
            stacked_othergrads, cov = self._apply_covariance(othergrads)
            preconditioner = tf.cholesky(tf.ones_like(cov) + eta * cov)
            for g in othergrads:
                #preconditioned_grad = preconditioned_grad + self.get_covariance_component(grad, g) * g
                preconditioned_grad = preconditioned_grad + preconditioner * stacked_othergrads
        return preconditioned_grad

    def _extract_grads(self, grads_and_vars, var):
        """ Helper function to extract the gradient associated with `var` from
        the list containing all gradients and variables in `grads_and_vars`.
        Moreover, we return also all other gradiets (in replicas) associated to
        the same var.

        :param grads_and_vars: list of grad, var tuples over all replica
        :param var: current variable
        :return: grad associated to `var`,
                other grads associated to equivalent var in other replica
        """
        #print(var.name[var.name.find("/"):])
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

    def _apply_covariance(self, othergrads):
        """ Returns node for the covariance between the gradients of all other
        replica plus the identity.

        :param othergrads: list of all other gradients
        :return: node for the stacked gradients and the covariance matrix
        """
        grads = tf.stack(othergrads)
        number_replica = len(othergrads)
        # store mean nodes for ref within cov
        means = []
        for i in range(number_replica):
            means.append(tf.reduce_mean(grads[:, i]))
        # calculate cov as matrix between grads
        dim = math_ops.cast(tf.size(othergrads[0]), dds_basetype)
        norm_factor = 1. / (dim - 1.)
        cov = []
        # fill lower triangular and diagonal part
        for i in range(number_replica):
            cov.append([])
            for j in range(i+1):
                cov[-1].append(norm_factor * tf.reduce_sum((grads[:, i] - means[i]) * (grads[:,j] - means[j])))
        # fill in symmetric upper triangular part
        for i in range(number_replica):
            for j in range(i + 1, number_replica):
                cov[i].append(cov[j][i])
        return grads, cov

    def _stack_gradients(self, grads_and_vars, var):
        """ Stacks all (other) gradients together to compute a covariance matrix.

        :param grads_and_vars: list of gradients and vars from all replica
        :param var: this replicas variable
        :return: stacked gradients from _other_ replica (excluding the one
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
        print(grads)
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
