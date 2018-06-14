import tensorflow as tf

a = tf.constant(0., shape=[2,2], dtype=tf.float64)

with tf.Session() as sess:
	print(sess.run(a))
	# this fails because slicing is only available for variables
	print(sess.run(a[1,1].assign(1.)))
	# this fails als (constant) Tensor has no assign
	print(sess.run(a.assign([[1.,1.],[1.,1.]])))
	print(sess.run(a))

