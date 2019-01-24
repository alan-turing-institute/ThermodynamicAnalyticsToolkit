#
#    ThermodynamicAnalyticsToolkit - analyze loss manifolds of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#    The TATi authors, see file AUTHORS, have asserted their moral rights.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
### 

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