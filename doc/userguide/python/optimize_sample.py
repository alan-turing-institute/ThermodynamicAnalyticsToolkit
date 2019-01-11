from TATi.model import Model as tati

import numpy as np

FLAGS = tati.setup_parameters(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    every_nth=10,
    hidden_dimension=[1],
    input_columns=["x1"],
    learning_rate=1e-2,
    max_steps=100,
    optimizer="GradientDescent",
    output_activation="linear",
    seed=426,
    step_width=1e-2,
    trajectory_file="trajectory.csv"
)
nn = tati(FLAGS)
nn.init_input_pipeline()
nn.init_network(None, setup="train")
nn.reset_dataset()
opt_run_info, opt_trajectory, _ = nn.train( \
    return_run_info=True, return_trajectories=True)

FLAGS.max_steps = 5000
FLAGS.fix_parameters = "layer1/biases/Variable:0=0.;output/biases/Variable:0=0."
FLAGS.friction_constant = 10.
FLAGS.inverse_temperature = .2
FLAGS.sampler = "BAOAB"
nn.reset_parameters(FLAGS)
nn.init_network(None, setup="sample")
nn.reset_dataset()

# redo the input pipeline and its dataset as max_steps has changed
# and this the number of internal copies of the dataset.
nn.init_input_pipeline()
nn.reset_dataset()

sample_run_info, sample_trajectory, _ = nn.sample( \
    return_run_info=True, return_trajectories=True)


print("Sample results")
print(np.asarray(sample_run_info[0][0:10]))
print(np.asarray(sample_trajectory[0][0:10]))
