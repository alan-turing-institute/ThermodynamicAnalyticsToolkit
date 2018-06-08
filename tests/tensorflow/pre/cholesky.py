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
