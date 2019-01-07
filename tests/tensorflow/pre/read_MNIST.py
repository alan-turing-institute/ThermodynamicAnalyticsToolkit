#
#    ThermodynamicAnalyticsToolkit - explore high-dimensional manifold of neural networks
#    Copyright (C) 2018 The University of Edinburgh
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

from TATi.models.model import model

import logging
import numpy as np
import sys

import tensorflow as tf

FLAGS = model.setup_parameters(
    batch_data_files=[sys.argv[1]],
    batch_data_file_type="tfrecord",
    batch_size=500,
    hidden_dimension=[10],
    hidden_activation="relu",
    input_dimension=784,
    loss="softmax_cross_entropy",
    max_steps=100,
    optimizer="GradientDescent",
    output_activation="relu",
    output_dimension=10,
    seed=426,
    step_width=1e-2
)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

print(FLAGS.input_columns)

# nn = model(FLAGS)
# nn.init_network(None, setup="train")
# nn.init_input_pipeline()
# nn.reset_dataset()
#
# nn.train(False, False, False)
#
# nn_parameters = np.append(nn.weights[0].evaluate(nn.sess),nn.biases[0].evaluate(nn.sess))
# np.save("parameters.npy", nn_parameters)
#
# sys.exit(0)

nn = model(FLAGS)
nn.init_input_pipeline()
nn.init_network(None, setup="none")
nn.reset_dataset()

# load parameters
nn_parameters = np.load(sys.argv[2])
# assign parameters of NN
# ... assign parameters e.g. through parameter update directly
# in the np array, then call ...
nn.assign_neural_network_parameters(nn_parameters)


i=0
continue_flag = True
while continue_flag:
	try:
		features, labels = nn.input_pipeline.next_batch(nn.sess,
				                                auto_reset=False)
	except tf.errors.OutOfRangeError:
		continue_flag = False

	feed_dict = {
	    nn.xinput: features,
	    nn.nn[0].placeholder_nodes["y_"]: labels}

	loss_eval, acc, yeval, y_eval = \
        nn.sess.run([nn.loss,
                     nn.nn[0].summary_nodes["accuracy"],
                     nn.nn[0].placeholder_nodes["y"],
                     nn.nn[0].placeholder_nodes["y_"]],
                    feed_dict=feed_dict)

	i+=1
	print(str(i)+": "+str(loss_eval)+", "+str(acc)+", "+str(np.linalg.norm(yeval-y_eval)/50))


