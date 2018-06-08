import tensorflow as tf

from TATi.models.model import model

import numpy as np
import sys

FLAGS = model.setup_parameters(
    batch_data_files=[sys.argv[1]],
    batch_size=500,
    max_steps=1000,
    output_activation="linear",
    sampler="GeometricLangevinAlgorithm_2ndOrder",
    seed=426,
    step_width=1e-2
)
print(FLAGS)

FLAGS2 = FLAGS
FLAGS2.seed = 427
print(FLAGS)
print(FLAGS2)


nn1 = model(FLAGS)
nn1.init_network(None, setup="sample")
nn1.init_input_pipeline()
run_info, trajectory, averages = nn1.sample(return_run_info=True, \
  return_trajectories=True, return_averages=True)
nn1.finish()

print("Sample results")
print(np.asarray(run_info[0:10]))
print(np.asarray(trajectory[0:10]))
print(np.asarray(averages[0:10]))

tf.reset_default_graph()

nn2 = model(FLAGS)
nn2.init_network(None, setup="sample")
nn2.init_input_pipeline()
run_info, trajectory, averages = nn2.sample(return_run_info=True, \
  return_trajectories=True, return_averages=True)
nn2.finish()


