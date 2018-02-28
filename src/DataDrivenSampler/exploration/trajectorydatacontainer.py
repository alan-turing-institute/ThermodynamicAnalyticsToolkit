from DataDrivenSampler.exploration.trajectorydata import TrajectoryData

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

    def get_data(self, _id):
        """ This returns the datum to a given id

        :param _id: id to retrieve data object for
        :return: data object or None if id not found
        """
        if _id in self.data.keys():
            return self.data[_id]
        else:
            return None
