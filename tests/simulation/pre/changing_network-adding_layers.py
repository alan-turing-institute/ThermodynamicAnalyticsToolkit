import numpy as np
import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    hidden_dimension=[2],
    number_walkers=1,
    seed=426,
    verbose=1,
)

# fix numpy seed, too, as new parameters depend on it
np.random.seed(426)

# print loss with network with no hidden layers
print("Loss: "+str(nn.loss()))
print("Parameters: "+str(nn.parameters))

# increase hidden layer nodes
nn.set_options(hidden_dimension=[3,2])

# print loss with network with hidden layers
print("Loss: "+str(nn.loss()))
print("Parameters: "+str(nn.parameters))

