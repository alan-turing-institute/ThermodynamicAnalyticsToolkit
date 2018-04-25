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

import collections

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
    def __init__(self, covariance_blending=0., use_locking=False, name='ReplicaOptimizer'):
        """ Init function for this class.

        :param covariance_blending: mixing for preconditioning matrix to gradient
                update, identity matrix plus this times the covariance matrix obtained
                from the other replica
        :param use_locking: whether to lock in the context of multi-threaded operations
        :param name: internal name of optimizer
        """
        super(ReplicaOptimizer, self).__init__(use_locking, name)
        self.covariance_blending = covariance_blending

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
        grad, _ = self._extract_grads(grads_and_vars, var)
        picked_var, flat_othervars = self._extract_vars(grads_and_vars, var)
        if len(flat_othervars) != 0:
            _, cov = self._apply_covariance(flat_othervars, var)
            # \sqrt{ 1_D + \eta cov(flat_othervars)}, see [Matthews, Weare, Leimkuhler, 2016]
            unity = tf.matrix_band_part(tf.ones_like(cov, dtype=dds_basetype), 0, 0)
            matrix = unity + self.covariance_blending * cov
            max_matrix = tf.reduce_max(tf.abs(matrix))
            def accept():
                return tf.reciprocal(max_matrix)
            def reject():
                return max_matrix
            normalizing_factor = tf.cond(tf.greater(max_matrix, 1.),
                                         accept, reject)
            preconditioner = tf.sqrt(tf.reciprocal(normalizing_factor)) * \
                             tf.cholesky(normalizing_factor * matrix)
            # apply the covariance matrix to the flattened gradient and then return to
            # original shape to match for variable update

            # matrix-vector multiplication is however a bit more complicated, see
            # https://stackoverflow.com/a/43285258/1967646
            preconditioned_grad = tf.reshape(
                tf.matmul(tf.expand_dims(tf.reshape(grad, [-1]), 0), preconditioner),
                                             grad.get_shape())
        else:
            preconditioned_grad = grad
        return preconditioned_grad

    def _extract_grads(self, grads_and_vars, var):
        """ Helper function to extract the gradient associated with `var` from
        the list containing all gradients and variables in `grads_and_vars`.
        Moreover, we return also all other gradients (in replicas) associated to
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

    def _extract_vars(self, grads_and_vars, var):
        """ Helper function to extract the variable associated with `var` from
        the list containing all gradients and variables in `grads_and_vars`.
        Moreover, we return also all other gradients (in replicas) associated to
        the same var.

        :param grads_and_vars: list of grad, var tuples over all replica
        :param var: current variable
        :return: other vars associated to equivalent var in other replica
        """
        #print(var.name[var.name.find("/"):])
        othervars = []
        for i in range(len(grads_and_vars)):
            for g, v in grads_and_vars[i]:
                if g is not None:
                    if v.name == var.name:
                        pass
                    elif v.name[v.name.find("/"):] == var.name[var.name.find("/"):]:
                        othervars.append(tf.reshape(v, [-1]))
        return var, othervars

    def _apply_covariance(self, flat_othervars, var):
        """ Returns node for the covariance between the variables of all other
        replica plus the identity.

        :param flat_othervars: list of all other variables
        :return: node for the stacked variables and the covariance matrix
        """
        # we need a flat vars tensor as otherwise the index arithmetics becomes
        # very complicate for the "matrix * vector" product (covariance matrix
        # times the gradient) in the end. This is undone in `_pick_grad()`
        vars = tf.stack(flat_othervars)
        number_dim = tf.size(flat_othervars[0])

        # means are needed for every component. Hence, we save a bit by
        # creating all means beforehand
        means = tf.Variable(tf.zeros_like(flat_othervars[0], dtype=dds_basetype),
                            trainable=False, dtype=dds_basetype, name="mean")

        def body_mean(i, mean_copy):
            with tf.control_dependencies([tf.Print(means[i].assign(
                    tf.reduce_mean(vars[:, i], name="reduce_mean"),
                    name="assign_mean_component"), [var.name, vars[:, i]], "vars for mean: ")]):
                return (tf.add(i, 1), mean_copy)

        i = tf.constant(0)
        c = lambda i, x: tf.less(i, number_dim)
        r, mean_eval = tf.while_loop(c, body_mean, (i, means), name="means_loop")
        means = tf.Print(mean_eval, [mean_eval.name, tf.size(mean_eval), mean_eval], "mean_eval: ")
        print(means)

        # complicated way of constructing a D \times D matrix when we cannot use
        # D directly and only have a D vector as template: use an outer product
        # with the known vector
        template = tf.zeros_like(flat_othervars[0], dtype=dds_basetype)
        expanded_template =  tf.expand_dims(template, -1)
        # Use tf.shape() to get the runtime size of `x` in the 0th dimension.
        cov = tf.Variable(
            expanded_template * tf.transpose(expanded_template), dtype=dds_basetype, name="covariance_bias")
        print(cov)

        # pair of indices for the covariance loop
        Pair = collections.namedtuple('Pair', 'i, j')

        # in the following we implement two nested loops to go through every
        # component of the covariance matrix. The inner loop uses an extra
        # tf.cond as I don't want to use anymore tf.while. There, the problem is
        # that the body needs to return exactly the loop_vars. With an additional
        # inner loop however, there is another loop var and the "structures" do
        # not match anymore.

        # only on cov we can use sliced assignment, cov_copy seems to be some
        # ref object (or other) which does not have this functionality

        # note that the assigns modifies directly the node cov which lives
        # outside the accept() function's scope.
        def body_cov(p, cov_copy):
            def accept():
                # implement the assign as side effect when i,j is still valid:
                # if we implement it in the scope of body_cov(), then it would
                # also be applied when (i,j) is not a valid index pair.

                # the return statements of both accept and reject need to have
                # an identical structure. Therefore, we always return the
                # # tuple (i,j).
                with tf.control_dependencies([cov[p.i, p.j].assign(
                                norm_factor * tf.reduce_sum(
                                    (vars[:, p.i] - means[p.i])
                                * (vars[:, p.j] - means[p.j]),
                            name="cov_sum_reduction"),
                        name="cov_assign_component")]):
                    #return Pair(tf.Print(tf.identity(p.i, name="id"), [p.i], "i: "),
                    #            tf.Print(tf.add(p.j, 1, name="j_add"), [p.j], "j: "))
                    return Pair(tf.identity(p.i, name="id"),
                                tf.add(p.j, 1, name="j_add"))

            def reject():
                # assign is not allowed, hence we use subtract to set to zero
                #return Pair(tf.Print(tf.add(p.i, 1, name="i_add"), [p.i], "i: "),
                #            tf.Print(tf.subtract(p.j, p.j, name="j_reset"), [p.j], "j: "))
                return Pair(tf.add(p.i, 1, name="i_add"),
                            tf.subtract(p.j, p.j, name="j_reset"))

            ci = tf.less(p.j, number_dim, name="inner_check")
            li = tf.cond(ci, accept, reject, name="inner_conditional")
            return (li, cov_copy)

        dim = math_ops.cast(number_dim, dtype=dds_basetype)

        def accept_block():
            return tf.reciprocal(dim - 1.)

        def reject_block():
            return tf.constant(1.)

        norm_factor = tf.cond(
            tf.greater(tf.size(number_dim), 1),
            accept_block, reject_block)

        p = Pair(tf.constant(0), tf.constant(0))
        c = lambda p, _: tf.less(p.i, number_dim, name="outer_check")
        r, cov = tf.while_loop(c, body_cov, loop_vars=(p, cov), name="cov_loop")

        # note that cov will trigger the whole loop and also the loop to obtain
        # the means because of the dependency graph inside tensorflow.

        return vars, tf.Print(cov, [cov.name, cov], "cov: ")

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
