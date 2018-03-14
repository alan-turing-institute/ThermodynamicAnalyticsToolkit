from DataDrivenSampler.models.model import model
from DataDrivenSampler.exploration.explorer import Explorer

import numpy as np

FLAGS = model.setup_parameters(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=500,
    diffusion_map_method="vanilla",
    max_steps=10,
    number_of_eigenvalues=1,
    optimizer="GradientDescent",
    output_activation="linear",
    sampler="BAOAB",
    seed=426,
    step_width=1e-2,
    use_reweighting=False
)
nn = model(FLAGS)
# init both sample and train right away
nn.init_network(None, setup="sample")
nn.init_network(None, setup="train")
nn.init_input_pipeline()

explorer = Explorer(max_legs=5, number_pruning=0)

print("Creating starting trajectory.")
# a. add three legs to queue
explorer.spawn_starting_trajectory(nn)
# b. continue until queue has run dry
explorer.run_all_jobs(nn, FLAGS)

print("Starting multiple explorations from starting trajectory.")
# 2. with the initial trajectory done and analyzed,
#    find maximally separate points and sample from these
max_exploration_steps = 2
exploration_step = 1
while exploration_step < max_exploration_steps:
    # a. combine all trajectories
    steps, parameters, losses = \
        explorer.combine_sampled_trajectories()
    # b. perform diffusion map analysis for eigenvectors
    idx_corner = \
        explorer.get_corner_points(parameters, losses, \
                                   FLAGS, \
                                   number_of_corner_points=1)
    # d. spawn new trajectories from these points
    explorer.spawn_corner_trajectories(steps, parameters, losses,
                                       idx_corner, nn)
    # d. run all trajectories till terminated
    explorer.run_all_jobs(nn, FLAGS)

    exploration_step += 1

nn.finish()

run_info, trajectory = explorer.get_run_info_and_trajectory()

print("Exploration results")
print(np.asarray(run_info[0:10]))
print(np.asarray(trajectory[0:10]))
