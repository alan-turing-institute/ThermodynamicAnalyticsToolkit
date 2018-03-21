class TrajectoryJobId(object):
    """ This class encapsulates a trajectory job id in order to have it semaphorable

    """

    def __init__(self, initial_id):
        self.current_id = initial_id

    def get_unique_id(self):
        returnid = self.current_id
        self.current_id += 1
        return returnid