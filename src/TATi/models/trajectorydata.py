#
#    ThermodynamicAnalyticsToolkit - explore high-dimensional manifold of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
### 

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
