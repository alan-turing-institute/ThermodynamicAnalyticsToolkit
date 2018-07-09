import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters-small.csv"],
    fix_parameters="output/biases/Variable:0=2.",
    max_steps=5,
    seed=426,
)

run_info, trajectory, _ = nn.fit()

print("Train results")
print(np.asarray(trajectory[0:5]))
