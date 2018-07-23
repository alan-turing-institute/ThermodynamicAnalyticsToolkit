import numpy as np
import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    number_walkers=2,
    seed=427,
)

# get copies of walkers
parameters_first = nn.parameters[0]
parameters_second = nn.parameters[1]

# get losses
loss_first = nn.loss(walker_index=0)
loss_second = nn.loss(walker_index=1)

# losses should be different due to different parameter sets
assert( loss_first != loss_second )

# get gradients
gradients_first = nn.gradients(walker_index=0)
gradients_second = nn.gradients(walker_index=1)

# now exchange parameter sets
nn.parameters[0] = parameters_second
nn.parameters[1] = parameters_first

# check that losses are the same only flipped
assert( loss_second == nn.loss(walker_index=0) )
assert( loss_first == nn.loss(walker_index=1) )

# check that gradients are the same only flipped
assert( (gradients_second == nn.gradients(walker_index=0)).all() )
assert( (gradients_first == nn.gradients(walker_index=1)).all() )

# train network
nn.fit()

print(nn.score()[0])

