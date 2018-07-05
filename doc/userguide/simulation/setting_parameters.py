import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    seed=427,
)
# the old parameters
print(nn.parameters)
# set all parameters to zero
parameters = np.zeros(nn.num_parameters())
print(parameters)
nn.parameters = parameters
print(nn.parameters)