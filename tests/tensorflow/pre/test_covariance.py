import tensorflow as tf
from tensorflow.python.ops import math_ops

number_replica=3
number_dim=7

grads = []
grads.append(tf.constant([1, 4, 2, 5, 6, 24, 15], dtype=tf.float64))
grads.append(tf.constant([8, 5, 4, 6, 2, 1, 1], dtype=tf.float64))
grads.append(tf.constant([19, 2, 4, 3, 2, 2, 1], dtype=tf.float64))
stacked_grads = tf.stack(grads)

means = []
for i in range(number_dim):
	means.append(tf.reduce_mean(stacked_grads[:, i]))

dim = math_ops.cast(tf.shape(stacked_grads)[0], tf.float64)
norm_factor = 1. / (dim - 1.)
cov = []
for i in range(number_dim):
	cov.append([])
	for j in range(i+1):
		cov[-1].append(norm_factor * tf.reduce_sum((stacked_grads[:, i] - means[i]) * (stacked_grads[:, j] - means[j])))

for i in range(number_dim):
	for j in range(i+1, number_dim):
		#cov[i].append(cov[j][i])
		cov[i].append(tf.constant(0.))

with tf.Session() as sess:
    print(sess.run(dim))
    cov_eval = sess.run(cov)
    print(cov_eval)


