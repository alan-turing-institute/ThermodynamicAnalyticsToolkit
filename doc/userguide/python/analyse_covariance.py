import numpy as np
import matplotlib

# use agg as backend to allow command-line use as well
matplotlib.use("agg")
import matplotlib.pyplot as plt

from TATi.analysis.parsedtrajectory import ParsedTrajectory
from TATi.analysis.covariance import Covariance

trajectory = ParsedTrajectory("trajectory.csv")

num_eigenvalues=2
cov = Covariance(trajectory)
cov.compute( \
    number_eigenvalues=num_eigenvalues)

# plot in eigenvector directions
x = np.matmul(trajectory.get_trajectory(), cov.vectors[:,0])
y = np.matmul(trajectory.get_trajectory(), cov.vectors[:,1])
plt.scatter(x,y, marker='o', c=-x)
plt.savefig('covariance.png', bbox_inches='tight')
#plt.show()
