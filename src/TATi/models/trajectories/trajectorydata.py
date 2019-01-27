#
#    ThermodynamicAnalyticsToolkit - analyze loss manifolds of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#    The TATi authors, see file AUTHORS, have asserted their moral rights.
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

import numpy as np
import pandas as pd

class TrajectoryData(object):
    """This class is a simple structure that combines three pandas dataframes
    with information on a trajectory.

    This class is used to simplify the access when there is just a single
    walker.

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

        if self.run_info is not None:
            self.run_info = TrajectoryData.to_numeric(self.run_info)
        if self.trajectory is not None:
            self.trajectory = TrajectoryData.to_numeric(self.trajectory)
        if self.averages is not None:
            self.averages = TrajectoryData.to_numeric(self.averages)

    @staticmethod
    def _to_numeric_single_df(df):
        df = df.apply(pd.to_numeric)
        for name in ['id', 'step', 'epoch']:
            if name in df.columns:
                df[[name]] = df[[name]].astype(np.int64)
        return df

    @staticmethod
    def to_numeric(df):
        """ Convert the dataframe obtained from sampling or training to the
        correct dtypes.

        Notes:
          This needs to be able to deal with lists of dataframes from multiple
           walkers as well.

        Args:
          df: either single or list of multiple DataFrame's.

        Returns:
          modified instance of single DataFrame or list thereof
        """
        if isinstance(df, list):
            for i in range(len(df)):
                df[i] = TrajectoryData._to_numeric_single_df(df[i])
            return df
        else:
            return TrajectoryData._to_numeric_single_df(df)
