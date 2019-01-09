from TATi.model import Model as tati

import numpy as np

FLAGS = tati.setup_parameters(
    batch_data_files=["dataset-twoclusters-small.csv"],
    fix_parameters="output/biases/Variable:0=2.",
    max_steps=5,
    seed=426,
)

nn = tati(FLAGS)
nn.init_input_pipeline()
nn.init_network(None, setup="train")
nn.reset_dataset()
run_info, trajectory, _ = nn.train(return_run_info=True, \
  return_trajectories=True)

print("Train results")
print(np.asarray(trajectory[0:5]))
