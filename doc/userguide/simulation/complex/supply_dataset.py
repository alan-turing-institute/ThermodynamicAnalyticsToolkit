import TATi.simulation as tati

import numpy as np
import pandas as pd

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    output_activation="linear",
)
# e.g. parse dataset from CSV file into pandas frame
input_dimension = 2
output_dimension = 1
parsed_csv = pd.read_csv("dataset-twoclusters-test.csv", \
                         sep=',', header=0)
# extract feature and label columns as numpy arrays
features = np.asarray(\
    parsed_csv.iloc[:, 0:input_dimension])
labels = np.asarray(\
    parsed_csv.iloc[:, \
        input_dimension:input_dimension \
                        + output_dimension])

# supply dataset (this creates the input layer)
nn.dataset = [features, labels]

# this has created the network, now set
# parameters obtained from optimization run
nn.parameters = np.array([2.42835492e-01, 2.40057245e-01, \
    2.66429665e-03])

# evaluate loss and accuracy
print("Loss: "+str(nn.loss()))
print("Accuracy: "+str(nn.score()))
