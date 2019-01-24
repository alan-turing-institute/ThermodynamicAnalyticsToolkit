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

SEED = 426

# create single random node
def prepare_graph_first():
    # prepare a random tensor node
    a=tf.random_normal(shape=[1], dtype=tf.float64, name="randomness")
    return a

# create a node before the random one (should change ids)
def prepare_graph_second():
    # prepare a random tensor node
    b=tf.constant(1, dtype=tf.float64)
    a=tf.random_normal(shape=[1], dtype=tf.float64, name="randomness")
    return a, b

# create random node as first (i.e. same id as first case)
def prepare_graph_third():
    # prepare a random tensor node
    a=tf.random_normal(shape=[1], dtype=tf.float64, name="randomness")
    b=tf.constant(1, dtype=tf.float64)+a
    return a, b

def set_seed(seed):
    # fix the global seed
    tf.set_random_seed(seed)

RUNS=10
### FIRST a run

firstrun_a = []

tf.reset_default_graph()
set_seed(SEED)
with tf.Session() as sess:
    a = prepare_graph_first()
    # get ten numbers
    for i in range(RUNS):
        firstrun_a.append( sess.run(a) )

### FIRST b run

firstrun_b = []

tf.reset_default_graph()
set_seed(SEED)
with tf.Session() as sess:
    a = prepare_graph_first()
    # get ten numbers
    for i in range(RUNS):
        firstrun_b.append( sess.run(a) )


### SECOND run

secondrun = []

tf.reset_default_graph()
set_seed(SEED)
with tf.Session() as sess:
    a, b = prepare_graph_second()
    # get ten numbers
    for i in range(RUNS):
        secondrun.append( sess.run([b, a]) )

### THIRD run

thirdrun = []

tf.reset_default_graph()
set_seed(SEED)
with tf.Session() as sess:
    a, b = prepare_graph_third()
    # get ten numbers
    for i in range(RUNS):
        thirdrun.append(sess.run([b, a]))

# compare runs on identical networks
status = True
for i in range(RUNS):
    print("Comparing %lg to %lg," % (firstrun_a[i], firstrun_b[i]))
    if firstrun_a[i] != firstrun_b[i]:
        status = False

if not status:
    sys.exit(255)

# compare runs on modified networks with identical id
status = True
for i in range(RUNS):
    print("Comparing %lg to %lg," % (firstrun_a[i], thirdrun[i][1]))
    if firstrun_a[i] != thirdrun[i][1]:
        status = False

if not status:
    sys.exit(255)

# compare run to modified network with different id
status = True
for i in range(RUNS):
    print("Comparing %lg to %lg," % (firstrun_a[i], secondrun[i][1]))
    if firstrun_a[i] != secondrun[i][1]:
        status = False

if status:
    sys.exit(0)
else:
    sys.exit(1)