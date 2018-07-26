class TrajectoryData(object):
    """ This class is a simple structure that combines three pandas dataframes
    with information on a trajectory.

    """

    def __init__(self, run_info=None, trajectory=None, averages=None):
        if isinstance(run_info, list) and len(run_info) == 1:
            self.run_info = run_info[0]
        else:
            self.run_info = run_info
        if isinstance(trajectory, list) and len(trajectory) == 1:
            self.trajectory = trajectory[0]
        else:
            self.trajectory = trajectory
        if isinstance(averages, list) and len(averages) == 1:
            self.averages = averages[0]
        else:
            self.averages = averages
