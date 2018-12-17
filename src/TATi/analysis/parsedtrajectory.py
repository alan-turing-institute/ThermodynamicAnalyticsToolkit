import numpy as np
import pandas as pd

import sys

class ParsedTrajectory(object):
    """ This class encapsulates a single or multiple trajectories
    parsed from file.

    """
    def __init__(self, filename, every_nth=1):
        """

        :param filename: trajectory filename to parse
        :param every_nth: only consider every nth step
    *    """
        print("Loading trajectory file")
        self.df_trajectory = pd.read_csv(filename, sep=',', header=0)
        self.every_nth = every_nth
        self.start = 0

    def add_drop_burnin(self, drop_burnin):
        """ Allows to exclude an initial selection of steps.

        :param drop_burnin: up to which value in step column to exclude
        :return:  True - excluded, False - drop_burnin is illegal
        """
        steps = self.df_trajectory.loc[:,['step']].values
        if (len(steps) > 1) and (drop_burnin >= steps[1]):
            if drop_burnin < steps[-1]:
                self.start = next(x[0] for x in enumerate(steps) if x[1] > drop_burnin)
            else:
                return False
        return True

    def get_steps(self):
        return self.df_trajectory.loc[self.start::self.every_nth,['step']].values

    def get_loss(self):
        return self.df_trajectory.loc[self.start::self.every_nth,['loss']].values

    def get_trajectory(self):
        index = -1
        index2 = -1
        if "weight0" in self.df_trajectory.columns:
            index = self.df_trajectory.columns.get_loc('weight0')
        if "bias0" in self.df_trajectory.columns:
            index2 = self.df_trajectory.columns.get_loc('bias0')
        if (index2 < index and index2 >=0) or (index == -1):
            index = index2
        return self.df_trajectory.iloc[self.start::self.every_nth,index:].values
