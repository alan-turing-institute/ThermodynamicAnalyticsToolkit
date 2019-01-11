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

A1 = tf.constant([2., 1.], tf.float64)
A2 = tf.constant([4., 3.], tf.float64)
A = tf.stack([A1, A2])

means = [tf.reduce_mean(A[:,i]) for i in range(2)]
other_means = [tf.reduce_mean(A[i,:]) for i in range(2)]

with tf.Session() as sess:
	print(sess.run(means))
	print(sess.run(other_means))

