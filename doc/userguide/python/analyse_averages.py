from TATi.analysis.parsedtrajectory import ParsedTrajectory
from TATi.analysis.averagetrajectorywriter import AverageTrajectoryWriter

trajectory = ParsedTrajectory("trajectory.csv")
avg = AverageTrajectoryWriter(trajectory)
steps = trajectory.get_steps()

print(avg.average_params)
print(avg.variance_params)
