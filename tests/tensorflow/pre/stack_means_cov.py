import tensorflow as tf

A1 = tf.constant([2., 1.], tf.float64)
A2 = tf.constant([4., 3.], tf.float64)
A = tf.stack([A1, A2])

means = [tf.reduce_mean(A[:,i]) for i in range(2)]
other_means = [tf.reduce_mean(A[i,:]) for i in range(2)]

with tf.Session() as sess:
	print(sess.run(means))
	print(sess.run(other_means))

