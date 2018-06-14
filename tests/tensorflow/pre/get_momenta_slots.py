from TATi.models.model import model
import sys
import tensorflow as tf

FLAGS = model.setup_parameters( \
	batch_data_files=[sys.argv[1]],
	hidden_dimension="6 6")
nn = model(FLAGS)
nn.init_network(setup="sample")
momenta = tf.get_collection("slots")
print(momenta)

