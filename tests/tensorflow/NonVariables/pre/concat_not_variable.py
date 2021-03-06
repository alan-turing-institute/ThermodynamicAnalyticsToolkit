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

a=tf.Variable(tf.random_uniform([2], dtype=tf.float32))
b=tf.Variable(tf.random_uniform([2], dtype=tf.float32))
c=tf.concat([a,b], axis=0)

d_all=tf.placeholder(shape=[4], dtype=tf.float32)
d_single=tf.placeholder(shape=[2], dtype=tf.float32)

e_all=tf.assign(c,d_all)
e_single=tf.assign(a,d_single)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

print(a)
print(d_single)

sess.run(e_single, feed_dict={
    d_single: [1,2]
})

print(c)
print(d_all)


sess.run(e_all, feed_dict={
    d_all: [1,2,3,4]
})