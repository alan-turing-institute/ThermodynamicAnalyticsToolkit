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

accept_seed=426

p_accept = tf.constant(0.5, dtype=tf.float64)
uniform_random_t = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float64, seed=accept_seed)

zero_t = tf.constant(0., dtype=tf.float64)
one_t = tf.constant(1., dtype=tf.float64)
two_t = tf.constant(2., dtype=tf.float64)

total_energy_t = tf.Variable(1., dtype=tf.float64)

def true_fn():
	return total_energy_t.assign(zero_t)
def false_fn():
	return total_energy_t.assign(one_t)

accept_energy = total_energy_t.assign(zero_t)
reject_energy = total_energy_t.assign(one_t)

def not_energy():
	return total_energy_t.assign(two_t)

test_t = tf.greater(p_accept, uniform_random_t)

def accept_reject_block_t():
	return tf.cond(test_t,true_fn, false_fn)

step_width_t = tf.placeholder(name="step_width", dtype=tf.int32)
step_t = tf.cond(tf.equal(tf.mod(step_width_t,2),0), not_energy, accept_reject_block_t)

sess=tf.Session()

sess.run([tf.global_variables_initializer(),
          tf.local_variables_initializer()]
)

for i in range(10):
	total_energy = sess.run(step_t, feed_dict={
		step_width_t: i
	})
	print(total_energy)
