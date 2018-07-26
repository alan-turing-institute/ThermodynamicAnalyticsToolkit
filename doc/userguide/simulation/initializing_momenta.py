import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    seed=427,
)

np.random.seed(nn._options.seed)
# the old momenta
print(nn.momenta)
# reinitialize all momenta
nn.init_momenta(inverse_temperature=10)
print(nn.momenta)
