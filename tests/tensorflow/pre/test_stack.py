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

m = tf.constant([1.,2.])
print(m)
m_s = tf.stack([m,m])

with tf.Session() as sess:
	print(sess.run(m))
	print(sess.run(tf.reduce_mean(tf.expand_dims(m,[-1])[0, :])))
	print(sess.run(m_s[:,1]))
	print(sess.run(tf.reduce_mean(m_s[:, 1])))
