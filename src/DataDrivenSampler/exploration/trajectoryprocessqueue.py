import logging
import tempfile
import os, shutil

from multiprocessing.context import Process
from multiprocessing import JoinableQueue

from DataDrivenSampler.exploration.trajectoryprocess_sample import TrajectoryProcess_sample
from DataDrivenSampler.exploration.trajectoryprocess_train import TrajectoryProcess_train
from DataDrivenSampler.exploration.trajectoryjobqueue import TrajectoryJobQueue

class TrajectoryProcessQueue(TrajectoryJobQueue):
    ''' This class is a queue of trajectory jobs (of type run and analyze)
    which are executed in a FIFO fashion until the queue is empty.

    '''

    MAX_MINIMA_CANDIDATES = 3 # dont check more than this number of minima candidates

    def __init__(self, parameters, number_pruning, number_processes, manager):
        """ Initializes a queue of trajectory jobs.

        :param max_legs: maximum number of legs (of length max_steps) per trajectory
        :param number_pruning: number of pruning jobs added at trajectory end
        :param number_processes: number of concurrent processes to use
        :param manager: manager for semaphored instances used in multiprocessing
        """
        super(TrajectoryProcessQueue, self).__init__(parameters.max_legs, number_pruning, number_processes)
        self.parameters = parameters
        self.data_container =  manager.TrajectoryDataContainer()
        self.current_job_id = manager.TrajectoryJobId(1)
        self.queue = JoinableQueue()
        self.lock = manager.Lock()

    def getUniqueFilename(self, prefix="", suffix=""):
        """ Returns a unique filename

        :param prefix: prefix for temporary filename
        :param suffix: suffix for temporary filename
        :return: unique filename
        """
        if prefix == "":
            f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix)
        else:
            f = tempfile.NamedTemporaryFile(mode="w", prefix=prefix, suffix=suffix)
        name = f.name
        f.close()
        return name

    def add_sample_job(self, data_object, run_object=None, continue_flag=False):
        """ Adds a run job to the queue.

        :param data_object: data object for the job
        :param restore_model_filename: file name from where to restore model
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        data_object = self.instantiate_data_object(data_object, type="sample")
        temp_filenames = [ self.getUniqueFilename(prefix="run-", suffix=".csv"),
                           self.getUniqueFilename(prefix="trajectory-", suffix=".csv"),
                           self.getUniqueFilename(prefix="averages-", suffix=".csv")]
        if data_object.model_filename is None:
            data_object.model_filename = self.getUniqueFilename(prefix="model-")
            restore_model_filename = None
        else:
            restore_model_filename = data_object.model_filename+"/model"
        save_model_filename = data_object.model_filename+"/model"
        self.data_container.update_data(data_object)
        sample_job = TrajectoryProcess_sample(data_id=data_object.get_id(),
                                              network_model=run_object,
                                              lock=self.lock,
                                              FLAGS=self.parameters,
                                              temp_filenames=temp_filenames,
                                              restore_model=restore_model_filename,
                                              save_model=save_model_filename,
                                              continue_flag=continue_flag)
        self._enqueue_job(sample_job)

    def add_train_job(self, data_object, run_object=None, continue_flag=False):
        """ Adds a run job to the queue.

        :param data_object: data object for the job
        :param restore_model_filename: file name from where to restore model
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        data_object = self.instantiate_data_object(data_object, type="train")
        temp_filenames = [ self.getUniqueFilename(prefix="run-", suffix=".csv"),
                           self.getUniqueFilename(prefix="trajectory-", suffix=".csv"),
                           self.getUniqueFilename(prefix="averages-", suffix=".csv")]
        if data_object.model_filename is None:
            data_object.model_filename = self.getUniqueFilename(prefix="model-")
        restore_model_filename = data_object.model_filename + "/model"
        save_model_filename = data_object.model_filename+"/model"
        self.data_container.update_data(data_object)
        train_job = TrajectoryProcess_train(data_id=data_object.get_id(),
                                            network_model=run_object,
                                            lock=self.lock,
                                            FLAGS=self.parameters,
                                            temp_filenames=temp_filenames,
                                            restore_model=restore_model_filename,
                                            save_model=save_model_filename,
                                            continue_flag=continue_flag)
        self._enqueue_job(train_job)

    def trajectory_ended(self, data_id):
        """ Provides a hook for derived classes to do something when a trajectory
        is terminated.

        :param data_id: data id to this trajectory
        """
        # remove the model files when trajectory is done
        data_object = self.data_container.get_data(data_id)
        if os.path.isdir(data_object.model_filename):
            logging.debug("Removing "+data_object.model_filename)
            shutil.rmtree(data_object.model_filename)

    def run_next_job_till_queue_empty(self, network_model, parameters):
        """ Run jobs in queue till empty

        :param network_model:
        :param parameters:
        :return:
        """
        while True:
            self.run_next_job(network_model, parameters)
            self.queue.task_done()
        print("QUEUE IS EMPTY, process stopping.")

    def run_all_jobs(self, network_model, parameters):
        """ Run all jobs using a set of processes.

        :param network_model:
        :param parameters:
        :return:
        """
        processes = [Process(target=self.run_next_job_till_queue_empty, args=(network_model, parameters,)) \
                     for i in range(self.number_processes)]
        logging.info("Starting "+str(len(processes))+" processes.")
        for p in processes:
            p.start()
        self.queue.join()
        assert( len(self.used_data_ids.copy()) == 0 )
        for p in processes:
            p.terminate()
        for p in processes:
            p.join()
        running = True
        while running:
                running = any([p.is_alive() for p in processes])
        processes.clear()
        self.queue.close()
        # reset queue: otherwise we cannot push new items onto it
        self.queue = JoinableQueue()
