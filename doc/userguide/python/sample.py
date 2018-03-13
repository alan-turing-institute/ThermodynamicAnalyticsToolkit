from DataDrivenSampler.models.model import model

import numpy as np

FLAGS = model.setup_parameters(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    max_steps=1000,
    output_activation="linear",
    sampler="GeometricLangevinAlgorithm_2ndOrder",
    seed=426,
    step_width=1e-2
)
nn = model(FLAGS)
nn.init_network(None, setup="sample")
nn.init_input_pipeline()
run_info, trajectory, averages = nn.sample(return_run_info=True, \
  return_trajectories=True, return_averages=True)
nn.finish()

print("Sample results")
print(np.asarray(run_info[0:10]))
print(np.asarray(trajectory[0:10]))
print(np.asarray(averages[0:10]))
