from DataDrivenSampler.models.model import model

import numpy as np

FLAGS = model.create_mock_flags(
    batch_size=500,
    data_type=2,
    dimension=500,
    max_steps=1000,
    noise=0.1,
    output_activation="linear",
    sampler="GeometricLangevinAlgorithm_2ndOrder",
    seed=426,
    step_width=1e-2
)
nn = model(FLAGS)
nn.init_network(None, setup="sample")
run_info, trajectory = nn.sample(return_run_info=True, \
  return_trajectories=True)
nn.finish()

print("Sample results")
print(np.asarray(run_info[0:10]))
print(np.asarray(trajectory[0:10]))