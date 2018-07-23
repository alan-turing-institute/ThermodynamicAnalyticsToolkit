import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters-small.csv"],
    fix_parameters="output/biases/Variable:0=2.",
    max_steps=5,
    seed=426,
)

training_data = nn.fit()

print("Train results")
print(np.asarray(training_data.trajectory[0:5]))
