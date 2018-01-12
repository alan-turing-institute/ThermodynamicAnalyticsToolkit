import tensorflow as tf

from DataDrivenSampler.models.neuralnet_parameters import neuralnet_parameters
from DataDrivenSampler.exploration.trajectoryjob_analyze import TrajectoryJob_analyze
from DataDrivenSampler.exploration.trajectoryjob_run import TrajectoryJob_run
from collections import deque

class TrajectoryQueue(object):
    ''' This class is a queue of trajectory jobs (of type run and analyze)
    which are executed in a FIFO fashion until the queue is empty.

    '''
    def __init__(self, _data_container):
        """ Initializes a queue of trajectory jobs.

        :param _data_container: data container with all data object
        """
        self.data_container =  _data_container
        self.current_job_id = 1
        self.queue = deque()

    def add_analyze_job(self, data_id, continue_flag):
        """ Adds an analyze.

        :param data_id: id associated with data object for the job
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        analyze_job = TrajectoryJob_analyze(data_id=data_id,
                                            continue_flag=continue_flag)
        self._enqueue_job(analyze_job)

    def add_run_job(self, data_id, continue_flag):
        """ Adds a run_job.

        :param _data_id: id associated with data object for the job
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        run_job = TrajectoryJob_run(data_id=data_id,
                                    continue_flag=continue_flag)
        self._enqueue_job(run_job)

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

    def run_next_job(self, _run_object):
        ''' Takes the next job from the start of the queue and runs it.
        Will add new jobs to queue depending on the result of the run job.

        :param _run_object: external resource needed for the job to run
        '''
        current_job = self.queue.popleft()
        print("Current job #"+str(current_job.get_job_id())+": "+current_job.job_type)
        data_id = current_job.get_data_id()
        data_object = self.data_container.get_data(data_id)
        updated_data, continue_flag = current_job.run(data_object, _run_object)
        self.data_container.update_data(updated_data)
        if current_job.job_type == "run":
            self.add_analyze_job(data_id, current_job.continue_flag)
        elif continue_flag and (current_job.job_type == "analyze"):
            self.add_run_job(data_id, current_job.continue_flag)
        else:
            print("Unknown job type")
            assert( False )

    def is_empty(self):
        """ Returns whether the queue is empty

        :return: True - queue is empty, False not
        """
        return len(self.queue) == 0
