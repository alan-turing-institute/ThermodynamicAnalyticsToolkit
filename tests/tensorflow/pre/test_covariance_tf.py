#
#    ThermodynamicAnalyticsToolkit - explore high-dimensional manifold of neural networks
#    Copyright (C) 2018 The University of Edinburgh
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
from tensorflow.python.ops import math_ops
import collections

number_replica=3
number_dim=7

grads = []
grads.append(tf.constant([1, 4, 2, 5, 6, 24, 15], dtype=tf.float64))
grads.append(tf.constant([8, 5, 4, 6, 2, 1, 1], dtype=tf.float64))
grads.append(tf.constant([19, 2, 4, 3, 2, 2, 1], dtype=tf.float64))
stacked_grads = tf.stack(grads)

means = tf.Variable(tf.zeros(number_dim, dtype=tf.float64), trainable=False)
i = tf.constant(0)
c = lambda i, x: tf.less(i, number_dim)
b = lambda i, x: (tf.add(i, 1), means[i].assign(tf.reduce_mean(stacked_grads[:, i], name="reduce_mean"), name="assign_mean_component"))
r, mean_eval = tf.while_loop(c, b, (i, means), name="means_loop")

#with tf.Session() as sess:
#	sess.run([tf.global_variables_initializer(),
#                  tf.local_variables_initializer()]
#        )
#	print(sess.run([r]))
#	print(sess.run(mean_eval))

Pair = collections.namedtuple('Pair', 'i, j')

cov = tf.Variable(tf.zeros([number_dim, number_dim], dtype=tf.float64), trainable=False)

def b_cov(p, cov_ref):
	# one less because body is still executed even when past threshold
	ci = tf.less(p.j, number_dim-1, name="inner_check")
	#accept = lambda: Pair(tf.Print(tf.identity(p.i, name="id"), [p.i], "i: "), tf.Print(tf.add(p.j,1, name="j_add"), [p.j], "j: "))
	#reject = lambda: Pair(tf.Print(tf.add(p.i, 1, name="i_add"),[p.i], "i: "), tf.Print(tf.subtract(p.j, p.j, name="j_reset"), [p.j], "j: "))
	accept = lambda: Pair(tf.identity(p.i, name="id"), tf.add(p.j,1, name="j_add"))
	reject = lambda: Pair(tf.add(p.i, 1, name="i_add"), tf.subtract(p.j, p.j, name="j_reset"))
	li = tf.cond(ci, accept, reject, name="inner_conditional")
	i=p.i
	j=p.j
	cov_assign = cov[i,j].assign(norm_factor * tf.reduce_sum((stacked_grads[:, p.i] - mean_eval[p.i]) * (stacked_grads[:, p.j] - mean_eval[p.j]), name="cov_sum_reduction"), name="cov_assign_component") 
	return (li, cov_assign)

dim = math_ops.cast(tf.shape(stacked_grads)[1], tf.float64)
norm_factor = 1. / (dim - 1.)
p = Pair(tf.constant(0), tf.constant(0))
c = lambda p, cov: tf.less(p.i, number_dim, name="outer_check")
r, out = tf.while_loop(c, b_cov, loop_vars=(p, cov), name="cov_loop")

with tf.Session() as sess:
	sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()]
        )
	print(sess.run([r]))
	print(sess.run(out))
