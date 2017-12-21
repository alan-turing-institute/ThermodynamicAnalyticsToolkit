from DataDrivenSampler.models.model import model

import numpy as np

FLAGS = model.setup_parameters(
    batch_data_files=["dataset-twoclusters-small.csv"],
    fix_parameters="output/biases/Variable:0=2.",
    max_steps=5,
    seed=426,
)

nn = model(FLAGS)
nn.init_network(None, setup="train")
run_info, trajectory = nn.train(return_run_info=True, \
  return_trajectories=True)
nn.finish()

print("Train results")
print(np.asarray(trajectory[0:5]))
