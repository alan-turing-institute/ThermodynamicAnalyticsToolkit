import tensorflow as tf

m = tf.constant([1.,2.])
print(m)
m_s = tf.stack([m,m])

with tf.Session() as sess:
	print(sess.run(m))
	print(sess.run(tf.reduce_mean(tf.expand_dims(m,[-1])[0, :])))
	print(sess.run(m_s[:,1]))
	print(sess.run(tf.reduce_mean(m_s[:, 1])))
