import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    every_nth=100,
    fix_parameters="layer1/biases/Variable:0=0.;output/biases/Variable:0=0.",
    hidden_dimension=[1],
    input_columns=["x1"],
    learning_rate=1e-2,
    max_steps=100,
    optimizer="GradientDescent",
    output_activation="linear",
    sampler = "BAOAB",
    prior_factor=2.,
    prior_lower_boundary=-2.,
    prior_power=2.,
    prior_upper_boundary=2.,
    seed=428,
    step_width=1e-2,
    trajectory_file="trajectory.csv",
)
training_data = nn.fit()

nn.set_options(
    friction_constant = 10.,
    inverse_temperature = .2,
    max_steps = 5000,
)

sampling_data = nn.sample()

print("Sample results")
print(sampling_data.run_info[0:10])
print(sampling_data.trajectory[0:10])
