import pandas as pd
import numpy as np
import matplotlib
# use agg as backend to allow command-line use as well
matplotlib.use("agg")
import matplotlib.pyplot as plt

from TATi.models.model import model

FLAGS = model.setup_parameters(
    trajectory_file="trajectory.csv"
)

df_trajectory = pd.read_csv(FLAGS.trajectory_file, sep=',', \
    header=0)
traj=np.asarray(df_trajectory)

conv=np.zeros(traj.shape)

# then we plot the running averages of the parameters
# inside weights
for i in range(1,traj.shape[0]):
    for d in range(traj.shape[1]):
        
        conv[i,d]=np.mean(traj[:i,d])

[plt.scatter(range(len(traj)), conv[:,i]) \
    for i in range(traj.shape[1])]
plt.savefig('step-parameters.png', bbox_inches='tight')
#plt.show()

print(conv[-1,:])
