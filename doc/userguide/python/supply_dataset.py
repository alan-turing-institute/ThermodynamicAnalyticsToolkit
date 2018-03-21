from TATi.models.model import model

import numpy as np
import pandas as pd

FLAGS = model.setup_parameters(
    batch_size=500,
    max_steps=1000,
    output_activation="linear",
    sampler="GeometricLangevinAlgorithm_2ndOrder",
    seed=426,
    step_width=1e-2
)
# e.g. parse dataset from CSV file into pandas frame
input_dimension = 2
output_dimension = 1
parsed_csv = pd.read_csv("dataset-twoclusters.csv", \
                         sep=',', header=0)
# extract feature and label columns as numpy arrays
features = np.asarray(\
    parsed_csv.iloc[:, 0:input_dimension])
labels = np.asarray(\
    parsed_csv.iloc[:, \
        input_dimension:input_dimension \
                        + output_dimension])

# create network model
nn = model(FLAGS)
# supply dataset (this creates the input layer)
nn.provide_data(features, labels)
# create the network
nn.init_network(None, setup="sample")
nn.input_pipeline.reset(nn.sess)
# sample
run_info, trajectory, _ = nn.sample(return_run_info=True, \
  return_trajectories=True)
nn.finish()

print("Sample results")
print(np.asarray(run_info[0:10]))
print(np.asarray(trajectory[0:10]))
