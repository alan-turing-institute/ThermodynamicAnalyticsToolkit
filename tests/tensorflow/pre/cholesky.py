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

import numpy as np
import sys
import tensorflow as tf

A = np.array([[2,1],[1,2]])
print(A)
L = np.linalg.cholesky(A)
print(L)
LL = np.dot(L, L.T.conj())
print(LL)

A = tf.constant([[2., 1.], [1., 2.]], tf.float64)
A_diag = tf.matrix_band_part(A, 0, 0)
B = tf.cholesky(A)
#B_nodiag = B-tf.matrix_band_part(B, 0, 0)
#B_full = tf.transpose(B_nodiag)+B
BB = tf.tensordot(B, tf.conj(tf.transpose(B)), axes=1)

with tf.Session() as sess:
	B_eval = sess.run(B)
	print(B_eval)
	#print(sess.run(B_nodiag))
	#print(sess.run(B_full))
	BB_eval = sess.run(BB)
	print(BB_eval)

	if (LL != BB_eval).any():
		sys.exit(255)
