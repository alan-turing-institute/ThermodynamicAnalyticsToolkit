import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    number_walkers=2,
    seed=427,
)
# the old parameters
print(nn.parameters)
# set all parameters to zero
parameters = np.zeros(nn.num_parameters())
nn.parameters = parameters
print("Parameters are zero: "+str(nn.parameters))
# set parameters of first walker to [1,0,...]
parameters[0] = 1.
print(parameters)
nn.parameters[0] = parameters
print("Parameters 1st: "+str(nn.parameters))
# set parameters of second walker to [0,1,...]
parameters[1] = 1.
print(parameters)
nn.parameters[1] = parameters
print("Parameters 2nd: "+str(nn.parameters))
