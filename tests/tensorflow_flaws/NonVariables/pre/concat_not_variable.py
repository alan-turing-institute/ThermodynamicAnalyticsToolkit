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