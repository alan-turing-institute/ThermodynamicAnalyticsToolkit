import numpy as np
import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=10,
    seed=426,
)

# store current batch (this will trigger advance)
dataset = nn.dataset

# get loss
print(nn.loss())

# store again, this should not trigger
dataset_same = nn.dataset

# get loss, this should trigger advance
print(nn.loss())

# store again
dataset_different = nn.dataset

for k in range(2):
    assert( np.array_equal( dataset[k], dataset_same[k]))
    assert( not np.array_equal( dataset[k], dataset_different[k]))
    assert( not np.array_equal( dataset_same[k], dataset_different[k]))
