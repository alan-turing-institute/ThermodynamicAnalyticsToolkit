import tensorflow as tf

from DataDrivenSampler.models.neuralnet_parameters import neuralnet_parameters
from DataDrivenSampler.exploration.trajectoryjob_analyze import TrajectoryJob_analyze
from DataDrivenSampler.exploration.trajectoryjob_check_minima import TrajectoryJob_check_minima
from DataDrivenSampler.exploration.trajectoryjob_extract_minimum_candidates import TrajectoryJob_extract_minimium_candidates
from DataDrivenSampler.exploration.trajectoryjob_prune import TrajectoryJob_prune
from DataDrivenSampler.exploration.trajectoryjob_run import TrajectoryJob_run
from collections import deque

class TrajectoryQueue(object):
    ''' This class is a queue of trajectory jobs (of type run and analyze)
    which are executed in a FIFO fashion until the queue is empty.

    '''
    def __init__(self, _data_container, number_pruning):
        """ Initializes a queue of trajectory jobs.

        :param _data_container: data container with all data object
        :param number_pruning: number of pruning jobs added at trajectory end
        """
        self.data_container =  _data_container
        self.number_pruning = number_pruning
        self.current_job_id = 1
        self.queue = deque()

    def add_analyze_job(self, data_id, parameters, continue_flag):
        """ Adds an analyze job to the queue.

        :param data_id: id associated with data object for the job
        :param parameters: parameters for analysis
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        analyze_job = TrajectoryJob_analyze(data_id=data_id,
                                            parameters=parameters,
                                            continue_flag=continue_flag)
        self._enqueue_job(analyze_job)

    def add_check_minima_job(self, data_id, network_model, continue_flag):
        """ Adds a check_minima/optimize job to the queue.

        :param _data_id: id associated with data object for the job
        :param network_model: neural network object for running the graph
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        check_minima_job = TrajectoryJob_check_minima(data_id=data_id,
                                                      network_model=network_model,
                                                      continue_flag=continue_flag)
        self._enqueue_job(check_minima_job)

    def add_extract_minima_job(self, data_id, parameters, continue_flag):
        """ Adds an extract job to the queue.

        :param data_id: id associated with data object for the job
        :param parameters: parameters for analysis
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        extract_job = TrajectoryJob_extract_minimium_candidates(
            data_id=data_id,
            parameters=parameters,
            continue_flag=continue_flag)
        self._enqueue_job(extract_job)

    def add_prune_job(self, data_id, network_model, continue_flag):
        """ Adds a prune job to the queue.

        :param data_id: id associated with data object for the job
        :param network_model: neural network model
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        prune_job = TrajectoryJob_prune(data_id=data_id,
                                        network_model=network_model,
                                        continue_flag=False)
        self._enqueue_job(prune_job)

    def add_run_job(self, data_id, network_model, initial_step, parameters=None, continue_flag=False):
        """ Adds a run job to the queue.

        :param _data_id: id associated with data object for the job
        :param network_model: neural network object for running the graph
        :param initial_step: number of first step (for continuing a trajectory)
        :param parameters: parameters of the neural net to set. If None, keep random ones
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        run_job = TrajectoryJob_run(data_id=data_id,
                                    network_model=network_model,
                                    initial_step=initial_step,
                                    parameters=parameters,
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

    def run_next_job(self, run_object, analyze_object):
        ''' Takes the next job from the start of the queue and runs it.
        Will add new jobs to queue depending on the result of the run job.

        :param run_object: neural network object for run
        :param analyze_object: parameter object for analysis
        '''
        current_job = self.queue.popleft()
        print("Current job #"+str(current_job.get_job_id())+": "+current_job.job_type)
        data_id = current_job.get_data_id()
        data_object = self.data_container.get_data(data_id)
        updated_data, continue_flag = current_job.run(data_object)
        print("Continue? "+str(continue_flag))
        self.data_container.update_data(updated_data)
        if continue_flag and current_job.job_type == "run":
            print("Adding analyze job")
            self.add_analyze_job(data_id, analyze_object, current_job.continue_flag)
        elif current_job.job_type == "analyze":
            if continue_flag:
                print("Adding run job")
                self.add_run_job(data_id, run_object,
                                 data_object.legs_at_step[-1],
                                 data_object.parameters[-1],
                                 current_job.continue_flag)
            elif not updated_data.is_pruned:
                print("Adding prune job and post analysis")
                for i in range(self.number_pruning):
                    self.add_prune_job(data_id, run_object, False)
                self.add_extract_minima_job(data_id, analyze_object, False)
                if self.number_pruning == 0:
                    updated_data.is_pruned = True
                else:
                    self.add_analyze_job(data_id, analyze_object, False)
            else:
                if len(updated_data.minimum_candidates) > 0:
                    print("Adding check minima jobs")
                    self.add_check_minima_job(data_id, run_object, False)
                else:
                    print("No minimum candidates on this trajectory.")
        else:
            print("Not adding.")


    def is_empty(self):
        """ Returns whether the queue is empty

        :return: True - queue is empty, False not
        """
        return len(self.queue) == 0