import numpy as np
import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    number_walkers=2,
)

# check zeroing of all parameters
nn.parameters = np.zeros((nn.num_parameters()))
for k in range(nn._options.number_walkers):
    for i in range(nn.num_parameters()):
        assert( nn.parameters[k][i] == 0. )

# check single walker, multiple parameters access
nn.parameters = np.zeros((nn.num_parameters()))
nn.parameters[1] = np.ones((nn.num_parameters()))
assert( nn.parameters[1][0] == 1. )
assert( nn.parameters[1][1] == 1. )
assert( nn.parameters[0][0] == 0. )
assert( nn.parameters[0][1] == 0. )

# check multiple walker, multiple parameters access
nn.parameters = np.zeros((nn.num_parameters()))
nn.parameters = np.ones((nn.num_parameters()))
assert( nn.parameters[1][0] == 1. )
assert( nn.parameters[1][1] == 1. )
assert( nn.parameters[0][0] == 1. )
assert( nn.parameters[0][1] == 1. )
