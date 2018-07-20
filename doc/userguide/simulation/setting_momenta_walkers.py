import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    number_walkers=2,
    seed=427,
)
# the old momenta
print(nn.momenta)
# set all momenta to zero
momenta = np.zeros(nn.num_parameters())
nn.momenta = momenta
print("Momenta are zero: "+str(nn.momenta))
# set momenta of first walker to [1,0,...]
momenta[0] = 1.
print(momenta)
nn.momenta[0] = momenta
print("Momentum 1st: "+str(nn.momenta))
# set momenta of second walker to [0,1,...]
momenta[1] = 1.
print(momenta)
nn.momenta[1] = momenta
print("Momentum 2nd: "+str(nn.momenta))
