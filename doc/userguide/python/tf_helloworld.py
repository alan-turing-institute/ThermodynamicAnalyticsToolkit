import tensorflow as tf

a=tf.constant("Hello world")
sess=tf.Session()
print(sess.run(a))

