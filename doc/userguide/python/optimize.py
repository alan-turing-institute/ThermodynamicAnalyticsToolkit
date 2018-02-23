from DataDrivenSampler.models.model import model

import numpy as np

FLAGS = model.setup_parameters(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    max_steps=100,
    optimizer="GradientDescent",
    output_activation="linear",
    seed=426,
    step_width=1e-2
)
nn = model(FLAGS)
nn.init_network(None, setup="train")
nn.init_input_pipeline()
run_info, trajectory, _ = nn.train(return_run_info=True, \
  return_trajectories=True)
nn.finish()

print("Train results")
print(np.asarray(run_info[0:10]))
print(np.asarray(trajectory[0:10]))
