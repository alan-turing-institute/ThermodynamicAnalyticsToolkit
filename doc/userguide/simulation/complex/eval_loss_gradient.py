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

# alternativly, evaluate loss and gradients. otherwise loss and 
# gradient may not match due to different subset of dataset
# (if batch_size != dimension)
print(nn.gradients())

