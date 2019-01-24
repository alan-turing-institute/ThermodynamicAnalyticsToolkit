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

#!/usr/bin/env python3
#
# extracts the maximum diffusion length for a given diffusion map eigenvector
# file and adds it to an sql database

from TATi.model import Model as tati
from TATi.models.basetype import dds_basetype

import argparse
import sys
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--batch_data_files", type=str, default=None, \
	help="Input dataset file")
parser.add_argument("--hidden_dimension", type=int, nargs='+', default=[], \
	help="Column name speciyfing the eigenvector to inspect for maximum length")
parser.add_argument("--loss", type=str, default=None, \
	help="loss function to use")
parser.add_argument("--output_file", type=str, default=None, \
	help="Output CSV file")
parser.add_argument("--seed", type=int, default=None, \
	help="Seed for random starting configuration")
parser.add_argument('--version', '-V', action="store_true", \
	help='Gives version information')

params, _ = parser.parse_known_args()

if params.version:
	# give version and exit
	print(sys.argv[0]+" -- version 0.1")
	sys.exit(0)

# setup test pipeline
FLAGS = tati.setup_parameters(
    batch_data_files=[params.batch_data_files],
    batch_data_file_type="tfrecord",
    batch_size=100,
    every_nth=1,
    hidden_dimension=params.hidden_dimension,
    hidden_activation="relu",
    input_dimension=784,
    loss=params.loss,
    max_steps=50,
    optimizer="GradientDescent",
    output_activation="linear",
    output_dimension=10,
    seed=params.seed,
    step_width=5e-2
)
nn = tati(FLAGS)
nn.init_input_pipeline()
nn.init_network(None, setup="train")

test_pipeline = nn.input_pipeline
test_pipeline.reset(nn.sess)

# setup train pipeline
FLAGS.batch_size = 100
FLAGS.max_steps=50
nn.reset_parameters(FLAGS)
nn.init_input_pipeline()
nn.reset_dataset()

test_features, test_labels = test_pipeline.next_batch(nn.sess, auto_reset=True)
test_dict = {
    nn.xinput: test_features,
    nn.nn[0].placeholder_nodes["y_"]: test_labels}

with tf.variable_scope('var_walker1'):
	with tf.variable_scope("accumulate", reuse=True):
		gradients_t = tf.get_variable("gradients", dtype=dds_basetype)
		zero_gradients = gradients_t.assign(0.)
		virials_t = tf.get_variable("virials", dtype=dds_basetype)
		zero_virials = virials_t.assign(0.)

train_nodes = nn.nn[0].get_list_of_nodes(["train_step", "accuracy", "global_step","loss"])

with open(params.output_file, "w") as of:
	of.write("step,loss,accuracy,gradients\n")

	continue_flag = True
	while continue_flag:
		# train step
		try:
			features, labels = nn.input_pipeline.next_batch(nn.sess, auto_reset=False)
		except tf.errors.OutOfRangeError:
			continue_flag = False

		feed_dict = {
			nn.xinput: features,
			nn.nn[0].placeholder_nodes["y_"]: labels,
			nn.nn[0].placeholder_nodes["learning_rate"]: FLAGS.step_width,
			nn.nn[0].placeholder_nodes["keep_probability"]: 1.
		}

		# train step
		check_gradients, check_virials = nn.sess.run([zero_gradients, zero_virials])
		assert (abs(check_gradients) < 1e-10)
		assert (abs(check_virials) < 1e-10)
		_, acc, global_step, loss_eval = \
			nn.sess.run(train_nodes, feed_dict=feed_dict)

		# evaluate loss and gradients accurately
		loss_eval, acc_eval, grad_eval = \
			nn.sess.run([nn.loss,
						 nn.nn[0].summary_nodes["accuracy"],
						 gradients_t], feed_dict=test_dict)

		# store to file
		of.write(str(global_step)+","+str(loss_eval)+","+str(acc_eval)+","+str(grad_eval)+"\n")

