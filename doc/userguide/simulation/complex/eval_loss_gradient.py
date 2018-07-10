import TATi.simulation as tati

import numpy as np

# prepare parameters
nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=10,
    output_activation="linear"
)

# assign parameters of NN
nn.parameters = np.zeros([nn.num_parameters()])

# simply evaluate loss
print(nn.loss())

# also evaluate gradients (from same batch)
print(nn.gradients())



