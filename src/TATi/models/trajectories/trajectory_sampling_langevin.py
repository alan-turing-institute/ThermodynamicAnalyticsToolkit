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

import logging
import tensorflow as tf
import time

try:
    from tqdm import tqdm # allows progress bar
    tqdm_present = True
    # workaround: otherwise we get deadlock on exceptions,
    # see https://github.com/tqdm/tqdm/issues/469
    tqdm.monitor_interval = 0
except ImportError:
    tqdm_present = False

from TATi.models.trajectories.trajectory_sampling import TrajectorySampling

class TrajectorySamplingLangevin(TrajectorySampling):
    """This implements sampling of a trajectory using Langevin Dynamics."""
    def __init__(self, trajectory_state):
        super(TrajectorySamplingLangevin, self).__init__(trajectory_state)

    def get_placeholder_nodes(self):
        retlist = super(TrajectorySamplingLangevin, self).get_placeholder_nodes()
        retlist.extend( [self.state.nn[walker_index].get_dict_of_nodes(
            ["current_step"])
            for walker_index in range(self.state.FLAGS.number_walkers)])
        return retlist

    def update_feed_dict(self, feed_dict, placeholder_nodes, current_step):
        for walker_index in range(self.state.FLAGS.number_walkers):
            feed_dict.update({
                placeholder_nodes[walker_index]["current_step"]: current_step
            })
