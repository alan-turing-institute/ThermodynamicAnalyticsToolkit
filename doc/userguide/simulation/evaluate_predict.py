import TATi.simulation as tati
import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    seed=427,
)
np.random.seed(427)
other_features = np.random.random((2,2))
print(nn.predict(other_features))
