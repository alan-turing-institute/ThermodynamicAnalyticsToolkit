import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    seed=427,
)

np.random.seed(nn._options.seed)
# the old momenta
print(nn.momenta)
# set all momenta to normally distributed random values
momenta = np.random.standard_normal(size=(nn.num_parameters()))
print(momenta)
nn.momenta = momenta
print(nn.momenta)