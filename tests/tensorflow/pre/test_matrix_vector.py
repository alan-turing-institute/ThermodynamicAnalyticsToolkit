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

g = tf.constant([[1., 2.],[3.,4.]])
#m = tf.expand_dims(tf.constant([1.,1.]), [-1])
m = tf.Variable(tf.eye(num_rows=4,num_columns=4, dtype=tf.float32))
m = m[1,3].assign(1.)

g_exp = tf.expand_dims(tf.reshape(g, [-1]), 0)
mv = tf.matmul(m, g_exp, transpose_b=True)
m_g = tf.reshape(mv, g.get_shape())

with tf.Session() as sess:
	sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()]
        )
	print(sess.run(g))
	print(sess.run(m))
	print(sess.run(g_exp))
	print(sess.run(mv))
#	print(sess.run(mv_sq))
	print(sess.run(m_g))
