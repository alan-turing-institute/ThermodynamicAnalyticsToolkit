from collections import deque
import logging


class TrajectoryQueue(object):
    ''' This class is a queue of trajectory jobs of any type
    which are executed in a FIFO fashion until the queue is empty.

    '''

    def __init__(self, _data_container, max_legs, number_pruning):
        """ Initializes a queue of trajectory jobs.

        :param _data_container: data container with all data object
        :param .max_legs: maximum number of legs (of length max_steps) per trajectory
        :param number_pruning: number of pruning jobs added at trajectory end
        """
        self.data_container =  _data_container
        self.max_legs = max_legs
        self.number_pruning = number_pruning
        self.current_job_id = 1
        self.queue = deque()

    def _enqueue_job(self, _job):
        """ Adds a new job to the end of the queue, also giving it a unique
        job id.

        :param _job: job to add
        """
        _job.set_job_id(self.current_job_id)
        self.current_job_id += 1
        self.queue.append(_job)

    def remove_job(self, _job_id):
        ''' Removes a job of the given id from the queue.

        :param _job_id: id of the job
        :return true - job found and removed, false - job id not found
        '''
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

    def is_empty(self):
        """ Returns whether the queue is empty

        :return: True - queue is empty, False not
        """
        return len(self.queue) == 0
