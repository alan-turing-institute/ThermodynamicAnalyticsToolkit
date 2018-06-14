import tensorflow as tf

g = tf.constant([[1., 2.],[3.,4.]])
#m = tf.expand_dims(tf.constant([1.,1.]), [-1])
m = tf.Variable(tf.eye(num_rows=4,num_columns=4, dtype=tf.float32))
m = m[1,3].assign(1.)

g_exp = tf.expand_dims(tf.reshape(g, [-1]), 0)
mv = tf.matmul(m, g_exp, transpose_b=True)
m_g = tf.reshape(mv, g.get_shape())

with tf.Session() as sess:
	sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()]
        )
	print(sess.run(g))
	print(sess.run(m))
	print(sess.run(g_exp))
	print(sess.run(mv))
#	print(sess.run(mv_sq))
	print(sess.run(m_g))
