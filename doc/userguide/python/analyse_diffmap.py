import numpy as np
import pandas as pd
import matplotlib
# use agg as backend to allow command-line use as well
matplotlib.use("agg")
import matplotlib.pyplot as plt

from TATi.TrajectoryAnalyser import compute_diffusion_maps

# option values coming from the sampling
inverse_temperature=1e4

df_trajectory = pd.read_csv("trajectory.csv", sep=',',
    header=0)
traj=np.asarray(df_trajectory)

steps = df_trajectory.loc[0::, ['step']].values
loss = df_trajectory.loc[0::, ['loss']].values
# get index to first parameter column "weight0"
index = df_trajectory.columns.get_loc('weight0')
trajectory = df_trajectory.iloc[0::, index:].values

num_eigenvalues=2
vectors, values, q = compute_diffusion_maps(
    traj=trajectory,
    beta=inverse_temperature,
    loss=loss,
    nrOfFirstEigenVectors=num_eigenvalues,
    method="vanilla",
    use_reweighting=False)

plt.scatter(vectors[:,0], vectors[:,1])
plt.savefig('eigenvectors.png', bbox_inches='tight')
plt.show()
