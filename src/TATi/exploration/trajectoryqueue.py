import logging

class TrajectoryQueue(object):
    ''' This class is a queue of trajectory jobs of any type
    which are executed in a FIFO fashion until the queue is empty.

    '''

    def __init__(self, max_legs, number_pruning, number_processes=0):
        """ Initializes a queue of trajectory jobs.

        :param .max_legs: maximum number of legs (of length max_steps) per trajectory
        :param number_pruning: number of pruning jobs added at trajectory end
        """
        self.max_legs = max_legs
        self.number_pruning = number_pruning
        self.number_processes = number_processes
        self.data_container = None
        self.current_job_id = None
        self.queue = None
        self.used_data_ids = [] # make a simple list as default

    def get_data_container(self):
        return self.data_container

    def add_used_data_ids_list(self, _list):
        self.used_data_ids = _list

    def add_sample_job(self, data_object, run_object=None, continue_flag=False):
        """ Adds a sampling job to the queue.

        :param data_object: data object for the job
        :param restore_model_filename: file name from where to restore model
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        assert( False )

    def add_train_job(self, data_object, run_object=None, continue_flag=False):
        """ Adds a training job to the queue.

        :param data_object: data object for the job
        :param restore_model_filename: file name from where to restore model
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        assert( False )

    def instantiate_data_object(self, data_object, type="sample"):
        if data_object is None:
            data_id = self.data_container.add_empty_data(type=type)
            data_object = self.data_container.get_data(data_id)
        return data_object

    def _enqueue_job(self, _job):
        """ Adds a new job to the end of the queue, also giving it a unique
        job id.

        :param _job: job to add
        """
        _job.set_job_id(self.current_job_id.get_unique_id())
        if self.number_processes == 0:
            self.queue.append(_job)
        else:
            self.queue.put(_job)

    def remove_job(self, _job_id):
        ''' Removes a job of the given id from the queue.

        :param _job_id: id of the job
        :return true - job found and removed, false - job id not found
        '''
        if self.number_processes == 0:
            for job in list(self.queue):
                if job.get_job_id() == _job_id:
                    self.queue.remove(job)
                    return True
        return False

    def run_next_job(self, run_object, analyze_object):
        ''' Takes the next job from the start of the queue and runs it.
        Will add new jobs to queue depending on the result of the run job.

        :param run_object: neural network object for run
        :param analyze_object: parameter object for analysis
        '''
        pass

    def run_all_jobs(self, network_model, parameters):
        """ Run all jobs currently found in the Trajectory queue.

        :param network_model: model of neural network with Session for sample and optimize jobs
        :param parameters: parameter struct for analysis jobs
        """
        while not self.is_empty():
            self.run_next_job(network_model, parameters)

    def is_empty(self):
        """ Returns whether the queue is empty

        :return: True - queue is empty, False not
        """
        if self.number_processes == 0:
            return len(self.queue) == 0
        else:
            return self.queue.empty()
