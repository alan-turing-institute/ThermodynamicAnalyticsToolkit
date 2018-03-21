from TATi.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_check_gradient(TrajectoryJob):
    ''' This implements a job that checks the gradients of a given trajectory.

    '''

    GRADIENT_THRESHOLD = 1e-5 # threshold for accepting minimum

    def __init__(self, data_id, parameters, continue_flag = True):
        """ Initializes an analysis job.

        :param data_id: id associated with data object
        :param parameters: parameter for analysis
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_check_gradient, self).__init__(data_id)
        self.job_type = "check_gradient"
        self.parameters = parameters
        self.continue_flag = continue_flag

    def run(self, _data):
        """ This implements checking the gradient of a given trajectory stored
        in a data object.

        :param _data: data object to use
        :param _object: FLAGS object that contains sampling parameters
        :return: updated data object
        """
        # check whether last gradient is still larger than threshold
        return _data, (_data.gradients[-1] > self.GRADIENT_THRESHOLD) and self.continue_flag
