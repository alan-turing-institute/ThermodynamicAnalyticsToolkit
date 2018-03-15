import logging
import tempfile
from threading import Thread
from multiprocessing.pool import ThreadPool

from DataDrivenSampler.models.neuralnet_parameters import neuralnet_parameters
from DataDrivenSampler.exploration.trajectoryjob_analyze import TrajectoryJob_analyze
from DataDrivenSampler.exploration.trajectoryjob_check_gradient import TrajectoryJob_check_gradient
from DataDrivenSampler.exploration.trajectoryjob_extract_minimum_candidates import TrajectoryJob_extract_minimium_candidates
from DataDrivenSampler.exploration.trajectoryjob_prune import TrajectoryJob_prune
from DataDrivenSampler.exploration.trajectoryprocess_sample import TrajectoryProcess_sample
from DataDrivenSampler.exploration.trajectoryprocess_train import TrajectoryProcess_train
from DataDrivenSampler.exploration.trajectoryqueue import TrajectoryQueue

class TrajectoryProcessQueue(TrajectoryQueue):
    ''' This class is a queue of trajectory jobs (of type run and analyze)
    which are executed in a FIFO fashion until the queue is empty.

    '''

    MAX_MINIMA_CANDIDATES = 3 # dont check more than this number of minima candidates

    def __init__(self, _data_container, parameters, number_pruning, number_processes):
        """ Initializes a queue of trajectory jobs.

        :param _data_container: data container with all data object
        :param .max_legs: maximum number of legs (of length max_steps) per trajectory
        :param number_pruning: number of pruning jobs added at trajectory end
        :param number_processes: number of concurrent processes to use
        """
        super(TrajectoryProcessQueue, self).__init__(_data_container, parameters.max_legs, number_pruning)
        self.unique_filenames = {}  # stores used filenames per data id
        self.parameters = parameters
        self.number_processes = number_processes

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

    def add_check_gradient_job(self, data_id, parameters=None, continue_flag=True):
        """ Adds an check_gradient job to the queue.

        :param data_id: id associated with data object for the job
        :param parameters: parameters for analysis
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        check_gradient_job = TrajectoryJob_check_gradient(data_id=data_id,
                                                          parameters=parameters,
                                                          continue_flag=continue_flag)
        self._enqueue_job(check_gradient_job)

    def add_check_minima_jobs(self, data_id, network_model, continue_flag):
        """ Adds a check_minima/optimize job to the queue.

        :param _data_id: id associated with data object for the job
        :param network_model: neural network object for running the graph
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        data_object = self.data_container.get_data(data_id)
        assert( data_object is not None )
        # set parameters to ones from old leg (if exists)
        if len(data_object.minimum_candidates) > self.MAX_MINIMA_CANDIDATES:
            # sort the gradients
            gradient_at_minima_candidates = [data_object.gradients[i] for i in data_object.minimum_candidates]
            minima_indices = sorted(range(len(gradient_at_minima_candidates)), key=lambda k: gradient_at_minima_candidates[k])
            assert( gradient_at_minima_candidates[ minima_indices[0] ] <= gradient_at_minima_candidates[ minima_indices[self.MAX_MINIMA_CANDIDATES] ] )
            if abs(gradient_at_minima_candidates[ minima_indices[0] ] - gradient_at_minima_candidates[ minima_indices[self.MAX_MINIMA_CANDIDATES] ]) < 1e-10:
                same_gradient_index = 0
                for i in range(1,len(minima_indices)):
                    if abs(gradient_at_minima_candidates[ minima_indices[0] ] - gradient_at_minima_candidates[ minima_indices[i] ]) < 1e-10:
                        same_gradient_index = minima_indices[i]
                # pick MAX_MINIMA_CANDIDATES randomly.
                candidates = [data_object.minimum_candidates[i] for i in np.random.choice(
                    minima_indices[0:same_gradient_index],
                    size=self.MAX_MINIMA_CANDIDATES,
                    replace=False)]
                logging.info("Too many candidates with same lowest gradients, we look randomly at these: "+str(candidates))
            else:
                candidates = [data_object.minimum_candidates[i] for i in minima_indices[0:self.MAX_MINIMA_CANDIDATES]]
                logging.info("Too many candidates, we look at ones with lowest gradients: "+str(candidates))
        else:
            candidates = data_object.minimum_candidates
        for i in range (len(candidates)):
            minima_index = candidates[i]
            current_id = self.data_container.add_empty_data(type="train")
            new_data_object = self.data_container.data[current_id]
            new_data_object.steps[:] = [data_object.steps[minima_index]]
            new_data_object.parameters[:] = [data_object.parameters[minima_index]]
            new_data_object.losses[:] = [data_object.losses[minima_index]]
            new_data_object.gradients[:] = [data_object.gradients[minima_index]]
            self.add_train_job(
                data_id=current_id,
                network_model=network_model,
                initial_step=new_data_object.steps[-1],
                parameters=new_data_object.parameters[-1],
                continue_flag=continue_flag)

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

    def add_sample_job(self, data_id, restore_model_filename=None, continue_flag=False):
        """ Adds a run job to the queue.

        :param _data_id: id associated with data object for the job
        :param restore_model_filename: file name from where to restore model
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        temp_filenames = [ self.getUniqueFilename(prefix="run-", suffix=".csv"),
                           self.getUniqueFilename(prefix="trajectory-", suffix=".csv"),
                           self.getUniqueFilename(prefix="averages-", suffix=".csv")]
        save_model_filename = "model_"+self.getUniqueFilename()
        sample_job = TrajectoryProcess_sample(data_id=data_id,
                                              FLAGS=self.parameters,
                                              temp_filenames=temp_filenames,
                                              restore_model=restore_model_filename,
                                              save_model=save_model_filename,
                                              continue_flag=continue_flag)
        self._enqueue_job(sample_job)

    def add_train_job(self, data_id, restore_model_filename=None, continue_flag=False):
        """ Adds a run job to the queue.

        :param _data_id: id associated with data object for the job
        :param restore_model_filename: file name from where to restore model
        :param continue_flag: flag whether job should spawn more jobs or not
        """
        temp_filenames = [ self.getUniqueFilename(prefix="run-", suffix=".csv"),
                           self.getUniqueFilename(prefix="trajectory-", suffix=".csv"),
                           self.getUniqueFilename(prefix="averages-", suffix=".csv")]
        save_model_filename = "model_"+self.getUniqueFilename()
        train_job = TrajectoryProcess_train(data_id=data_id,
                                            FLAGS=self.parameters,
                                            temp_filenames=temp_filenames,
                                            restore_model=restore_model_filename,
                                            save_model=save_model_filename,
                                            continue_flag=continue_flag)
        self._enqueue_job(train_job)

    def run_next_job(self, run_object, analyze_object):
        ''' Takes the next job from the start of the queue and runs it.
        Will add new jobs to queue depending on the result of the run job.

        :param run_object: neural network object for run
        :param analyze_object: parameter object for analysis
        '''
        current_job = self.queue.popleft()
        data_id = current_job.get_data_id()
        data_object = self.data_container.get_data(data_id)
        logging.info("Current job #"+str(current_job.get_job_id())+": "+current_job.job_type)

        continue_flag = False
        updated_data = data_object
        if current_job.job_type in ["sample", "train"]:
            # create thread (or get from pool) and launch
            def work_function():
                global updated_data
                global continue_flag
                updated_data, continue_flag = current_job.run(data_object)

            thread = Thread(target = work_function, args = ())
            thread.start()
            thread.join()
        else:
            updated_data, continue_flag = current_job.run(data_object)

        logging.info("Continue? "+str(continue_flag))
        if continue_flag:
            if len(updated_data.legs_at_step) >= self.max_legs and \
                            current_job.job_type in ["analyze", "check_gradient"]:
                logging.info("Maximum number of legs exceeded, not adding any more jobs of type " \
                             +current_job.job_type+" for data id "+str(data_id)+".")
                if current_job.job_type == "analyze":
                    self.add_extract_minima_job(data_id, analyze_object, False)
                    logging.info("Added extract minima job")
            else:
                if current_job.job_type == "sample":
                    for i in range(self.number_pruning):
                        self.add_prune_job(data_id, run_object, False)
                    logging.info("Added prune job(s)")
                    self.add_analyze_job(data_id, analyze_object, current_job.continue_flag)
                    logging.info("Added analyze job")

                elif current_job.job_type == "train":
                    self.add_check_gradient_job(data_id)
                    logging.info("Added check gradient job")

                elif current_job.job_type == "check_gradient":
                    self.add_train_job(data_id=data_id,
                                       network_model=run_object,
                                       initial_step=data_object.legs_at_step[-1],
                                       parameters=data_object.parameters[-1],
                                       continue_flag=current_job.continue_flag)
                    logging.info("Added train job")

                elif current_job.job_type == "extract_minimium_candidates":
                    self.add_check_minima_jobs(data_id, run_object, True)
                    logging.info("Added check minima jobs")

                elif current_job.job_type == "analyze":
                    self.add_sample_job(data_id=data_id,
                                        network_model=run_object,
                                        initial_step=data_object.legs_at_step[-1],
                                        parameters=data_object.parameters[-1],
                                        continue_flag=current_job.continue_flag)
                    logging.info("Added sample job")

                else:
                    logging.warning("Unknown job type "+current_job.job_type+"!")
        else:
            if current_job.job_type == "analyze":
                self.add_extract_minima_job(data_id, analyze_object, False)
                logging.info("Added extract minima job")
            else:
                logging.info("Not adding.")
