from TATi.model import Model as tati

import numpy as np

FLAGS = tati.setup_parameters(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    max_steps=1000,
    output_activation="linear",
    sampler="GeometricLangevinAlgorithm_2ndOrder",
    seed=426,
    step_width=1e-2
)
nn = tati(FLAGS)
nn.init_input_pipeline()
nn.init_network(None, setup="sample")
nn.reset_dataset()
run_info, trajectory, averages = nn.sample(return_run_info=True, \
  return_trajectories=True, return_averages=True)

print("Sample results")
print(np.asarray(run_info[0][0:10]))
print(np.asarray(trajectory[0][0:10]))
print(np.asarray(averages[0][0:10]))
