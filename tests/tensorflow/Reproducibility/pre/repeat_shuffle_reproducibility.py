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

import numpy as np
import pandas as pd
import sys
import tensorflow as tf

csv_dataset = pd.read_csv(sys.argv[1], sep=',', header=0)
features = np.asarray(csv_dataset.iloc[:, 0:2])
labels = np.asarray(csv_dataset.iloc[:, 2:3])
assert (features.shape[0] == labels.shape[0])
dimension = features.shape[0]

features_placeholder = tf.placeholder(tf.float32, features.shape)
labels_placeholder = tf.placeholder(tf.float32, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

# this is the "wrong way round": first repeat, then shuffle
dataset = dataset.repeat(10)
dataset = dataset.shuffle(buffer_size=features.shape[0], seed=426)
dataset = dataset.batch(20)

iterator = dataset.make_initializable_iterator()
batch_next = iterator.get_next()

with tf.Session() as session:
    session.run(iterator.initializer,
                feed_dict={features_placeholder: features,
                           labels_placeholder: labels})

    vals = session.run(batch_next)
    print(vals)

    for i in range(features.shape[0]):
        for j in range(i,features.shape[0]):
            if i == j:
                continue
            if all((vals[0][i] == vals[0][j])):
                print("%d and %d rows have same entry." % (i,j))
                sys.exit(255)

sys.exit(0)

