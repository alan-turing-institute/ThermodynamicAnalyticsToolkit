import TATi.simulation as tati

import numpy as np
import pandas as pd

nn = tati(
    batch_data_files=["dataset_ho.csv"],
    batch_size=1,
    fix_parameters="output/biases/Variable:0=0.",
    friction_constant=10.0,
    input_dimension=1,
    inverse_temperature=1.0,
    loss="mean_squared",
    max_steps=1000,
    output_activation = "linear",
    output_dimension=1,
    run_file="run_ho.csv",
    sampler="BAOAB",
    seed=426,
    step_width=0.5,
    trajectory_file="trajectory_ho.csv",
    verbose=1
)

data = nn.sample()

df_trajectories = pd.DataFrame(data.trajectory)
# sampled trajectory
weight0 = np.asarray(df_trajectories['weight0'])
