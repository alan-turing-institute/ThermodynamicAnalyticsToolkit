import tensorflow as tf

t = tf.constant([1, 4, 2, 5, 6, 24, 15], dtype=tf.float64)

with tf.Session() as sess:
	print(sess.run(tf.size(t)))
