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

from TATi.exploration.trajectorydata import TrajectoryData

class TrajectoryDataContainer(object):
    ''' This class is a structure to contain all data associated with running
    and analysing a specific sampling trajectory such as parameters along the
    trajectory, losses and gradients, diffusion map eigenvectors and
    eigenvalues. The trajectory is split up into "legs" by which it is checked
    whether diffusion map values have converged. If this is the case, the
    trajectory is ended, pruned and pooled together with all other sampled
    values.

    '''
    def __init__(self):
        self.data = {}
        self.current_data_id = 1

    def add_empty_data(self, type="sample"):
        """ Adds a new data object to the container with an id unique to this
        container.

        :return: id of the new object
        """
        assert( self.current_data_id not in self.data.keys() )
        self.data[ self.current_data_id ] = TrajectoryData( self.current_data_id, type )
        return_id = self.current_data_id
        self.current_data_id += 1
        return return_id

    def get_ids(self):
        """ Returns a list of ids.

        :return: list of ids
        """
        return self.data.keys()

    def get_data(self, _id):
        """ This returns the datum to a given id

        :param _id: id to retrieve data object for
        :return: data object or None if id not found
        """
        if _id in self.data.keys():
            return self.data[_id]
        else:
            return None

    def update_data(self, data_object):
        """ Replace old data object by new data object

        :param data_object: new data object
        """
        data_id = data_object.get_id()
        assert( data_id in self.data.keys())
        del self.data[data_id]
        self.data[data_id] = data_object