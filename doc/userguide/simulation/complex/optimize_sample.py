import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    hidden_dimension=[1],
    input_columns=["x1"],
    learning_rate=1e-2,
    max_steps=100,
    optimizer="GradientDescent",
    output_activation="linear",
    sampler = "BAOAB",
    seed=426,
    step_width=1e-2,
)
opt_run_info, opt_trajectory, _ = nn.fit()

nn.set_options(
    friction_constant = 10.,
    inverse_temperature = 4.,
    max_steps = 1000,
)

sample_run_info, sample_trajectory, _ = nn.sample()

print("Sample results")
print(np.asarray(sample_run_info[0][0:10]))
print(np.asarray(sample_trajectory[0][0:10]))
