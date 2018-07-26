from TATi.models.model import model
import sys
import tensorflow as tf

FLAGS = model.setup_parameters( \
	batch_data_files=[sys.argv[1]],
	sampler="GeometricLangevinAlgorithm_2ndOrder",
	inverse_temperature=1e3,
	friction_constant=10.,
	step_width=1e-1,
	hidden_dimension=[6,6])
nn = model(FLAGS)
nn.init_network(setup="sample")

for v in tf.trainable_variables():
	print(nn.sampler[0].get_slot(v, "momentum"))

