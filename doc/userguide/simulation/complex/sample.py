import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    max_steps=1000,
    output_activation="linear",
    sampler="GeometricLangevinAlgorithm_2ndOrder",
    seed=426,
    step_width=1e-2
)
sampling_data = nn.sample()

print("Sample results")
print(np.asarray(sampling_data.run_info[0:10]))
print(np.asarray(sampling_data.trajectory[0:10]))
print(np.asarray(sampling_data.averages[0:10]))
