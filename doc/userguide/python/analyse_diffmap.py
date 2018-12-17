import matplotlib
import numpy as np
import pandas as pd

# use agg as backend to allow command-line use as well
matplotlib.use("agg")
import matplotlib.pyplot as plt

from TATi.analysis.parsedtrajectory import ParsedTrajectory
from TATi.analysis.diffusionmap import DiffusionMap

# option values coming from the sampling
inverse_temperature=2e-1

trajectory = ParsedTrajectory("trajectory.csv")

num_eigenvalues=2
dmap = DiffusionMap.from_parsedtrajectory(trajectory)
dmap.compute( \
    number_eigenvalues=num_eigenvalues,
    inverse_temperature=inverse_temperature,
    diffusion_map_method="vanilla",
    use_reweighting=False)

plt.scatter(trajectory.get_trajectory()[:,0], trajectory.get_trajectory()[:,1], c=dmap.vectors[:,0])
plt.savefig('eigenvectors.png', bbox_inches='tight')
#plt.show()
