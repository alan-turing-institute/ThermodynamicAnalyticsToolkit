class TrajectoryJob(object):
    ''' This is the base class for job object that can be placed in the
    TrajectoryQueue for processing.

    This class needs to be derived and a proper run() method set up and the
    type of the job set.

    '''

    def __init__(self, data_id):
        """ Initializes the trajectory job.

        :param _data_id: id associated with data object
        """
        self.data_id = data_id
        self.job_id = -1
        self.job_type = "generic"

    def set_job_id(self, job_id):
        """ Sets the job id for this job. Can only be done once.

        :param job_id: new id of the job
        """
        # assert its unset so far
        assert( self.job_id == -1 )
        self.job_id = job_id

    def get_data_id(self):
        """ Return the unique id of the associated data object.

        :return: unique id of data object
        """
        return self.data_id

    def get_job_id(self):
        """ Return the unique id of this job

        :return: unique id of object
        """
        return self.job_id

    def run(self, _data, _object=None):
        """ This function needs to overriden.

        :param _data: data object to use
        :param _object: additional run object
        :return: updated data object
        """
        assert( 0 )