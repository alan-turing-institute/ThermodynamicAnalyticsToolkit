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
    step_width=1e-2
)
run_info, trajectory, averages = nn.fit()

print("Train results")
print(np.asarray(run_info[-10::]))
print(np.asarray(trajectory[-10:]))
print(np.asarray(averages[-10:]))
