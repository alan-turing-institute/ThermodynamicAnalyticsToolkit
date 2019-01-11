import pandas as pd
import numpy as np
import matplotlib
# use agg as backend to allow command-line use as well
matplotlib.use("agg")
import matplotlib.pyplot as plt

from TATi.analysis.parsedtrajectory import ParsedTrajectory
from TATi.analysis.averagetrajectorywriter import AverageTrajectoryWriter

trajectory = ParsedTrajectory("trajectory.csv")
avg = AverageTrajectoryWriter(trajectory)
steps = trajectory.get_steps()

print(avg.average_params)
print(avg.variance_params)
