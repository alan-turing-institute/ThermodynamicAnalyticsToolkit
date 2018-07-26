import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

data = pd.read_csv("trajectory_ho.csv", sep=",", header=0)
trajectory = data.loc[:,'weight0']

#hist, bins = np.histogram(trajectory, bins=np.arange(20), density=True)

plt.hist(trajectory, bins='auto', density=True)  # arguments are passed to np.histogram
plt.title("Histogram with 'auto' bins")
plt.savefig("trajectory_ho.png")
#plt.show()
