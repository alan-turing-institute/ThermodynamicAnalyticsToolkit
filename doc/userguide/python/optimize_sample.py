from DataDrivenSampler.models.model import model

import numpy as np

FLAGS = model.setup_parameters(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    max_steps=100,
    optimizer="GradientDescent",
    output_activation="linear",
    seed=426,
    step_width=1e-2
)
nn = model(FLAGS)
nn.init_network(None, setup="train")
nn.init_input_pipeline()
opt_run_info, opt_trajectory, _ = nn.train( \
    return_run_info=True, return_trajectories=True)

FLAGS.max_steps = 1000
FLAGS.sampler = "GeometricLangevinAlgorithm_2ndOrder"
nn.reset_parameters(FLAGS)
nn.init_network(None, setup="sample")

# redo the input pipeline and its dataset as max_steps has changed
# and this the number of internal copies of the dataset.
nn.init_input_pipeline()

sample_run_info, sample_trajectory, _ = nn.sample( \
    return_run_info=True, return_trajectories=True)

nn.finish()

print("Sample results")
print(np.asarray(sample_run_info[0:10]))
print(np.asarray(sample_trajectory[0:10]))
