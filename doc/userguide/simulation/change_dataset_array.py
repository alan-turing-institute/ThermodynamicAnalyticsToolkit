import TATi.simulation as tati

import numpy as np
# create random (but fixes) 2d dataset, labels in {-1,1}
num_items = 2
np.random.seed(427)
features = np.random.random((num_items,2))
labels = np.random.random_integers(0,1, (num_items,1))*2-1

nn = tati()
nn.dataset = [features, labels]
print(nn.dataset)