import numpy as np
import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    number_walkers=1,
)

# print loss with network with no hidden layers
print("Parameters: "+str(nn.parameters))
print("Loss: "+str(nn.loss()))

# set a hidden layer with two nodes
nn.set_options(hidden_dimension=[2])

# this triggers ValueError at the moment

# print loss with network with hidden layers
print("Parameters: "+str(nn.parameters))
print("Loss: "+str(nn.loss()))

