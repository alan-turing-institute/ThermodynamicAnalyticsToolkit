import TATi.simulation as tati

import numpy as np

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    learning_rate=3e-2,
    max_steps=1000,
    optimizer="GradientDescent",
    output_activation="linear",
    seed=426,
)
training_data = nn.fit()

print("Train results")
print(np.asarray(training_data.run_info[-10::]))
print(np.asarray(training_data.trajectory[-10:]))
print(np.asarray(training_data.averages[-10:]))
