import pandas as pd
import numpy as np
import matplotlib
# use agg as backend to allow command-line use as well
matplotlib.use("agg")
import matplotlib.pyplot as plt

df_run = pd.read_csv("run.csv", sep=',', header=0)
run=np.asarray(df_run.loc[:,\
   ['step','loss','kinetic_energy', 'total_energy']])

plt.scatter(run[:,0], run[:,1])
plt.savefig('loss-step.png',
            bbox_inches='tight')
#plt.show()
