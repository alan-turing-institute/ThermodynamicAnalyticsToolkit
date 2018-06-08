import sys
import tensorflow as tf

# these numbers are taken from GLA1 Sampler Reproducibility test, step1
a=0.12056163
b=0.0966097414
c=0.2025543
d=0.142757

# two summation with different ordering
calc = tf.sqrt(tf.constant(a, tf.float32)+tf.constant(b, tf.float32)+tf.constant(c, tf.float32)+tf.constant(d, tf.float32))
calc2 = tf.sqrt(tf.constant(a, tf.float32)+tf.constant(c, tf.float32)+tf.constant(b, tf.float32)+tf.constant(d, tf.float32))
with tf.Session() as sess:
    result = sess.run(calc)
    #0.74998844
    result2 = sess.run(calc2)
    #0.7499885

    if result != result2:
        sys.exit(255)

sys.exit(0)