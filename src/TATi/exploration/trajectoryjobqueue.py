#
#    ThermodynamicAnalyticsToolkit - analyze loss manifolds of neural networks
#    Copyright (C) 2018 The University of Edinburgh
#    The TATi authors, see file AUTHORS, have asserted their moral rights.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
### 

from collections import deque
import logging

from TATi.exploration.trajectorydatacontainer import TrajectoryDataContainer
from TATi.exploration.trajectoryjob_analyze import TrajectoryJob_analyze
from TATi.exploration.trajectoryjob_check_gradient import TrajectoryJob_check_gradient
from TATi.exploration.trajectoryjob_extract_minimum_candidates import TrajectoryJob_extract_minimium_candidates
from TATi.exploration.trajectoryjob_prune import TrajectoryJob_prune
from TATi.exploration.trajectoryjob_sample import TrajectoryJob_sample
from TATi.exploration.trajectoryjob_train import TrajectoryJob_train
from TATi.exploration.trajectoryjobid import TrajectoryJobId
from TATi.exploration.trajectoryqueue import TrajectoryQueue

class TrajectoryJobQueue(TrajectoryQueue):
    """This class is a queue of trajectory jobs (of type run and analyze)
    which are executed in a FIFO fashion until the queue is empty.

    Args:

    Returns:

    """

    MAX_MINIMA_CANDIDATES = 3 # dont check more than this number of minima candidates

    def __init__(self, max_legs, number_pruning, number_processes=0):
        """Initializes a queue of trajectory jobs.

        Args:
          max_legs: maximum number of legs (of length max_steps) per trajectory
          number_pruning: number of pruning jobs added at trajectory end
          number_processes: number of concurrent processes to use (Default value = 0)

        Returns:

        """
        super(TrajectoryJobQueue, self).__init__(max_legs, number_pruning, number_processes)
        self.data_container =  TrajectoryDataContainer()
        self.current_job_id = TrajectoryJobId(1)
        self.queue = deque()

    def add_analyze_job(self, data_object, parameters, continue_flag):
        """Adds an analyze job to the queue.

        Args:
          data_object: data object for the job
          parameters: parameters for analysis
          continue_flag: flag whether job should spawn more jobs or not

        Returns:

        """
        analyze_job = TrajectoryJob_analyze(data_id=data_object.get_id(),
                                            parameters=parameters,
                                            continue_flag=continue_flag)
        self._enqueue_job(analyze_job)

    def add_check_gradient_job(self, data_object, parameters=None, continue_flag=True):
        """Adds an check_gradient job to the queue.

        Args:
          data_object: data object for the job
          parameters: parameters for analysis (Default value = None)
          continue_flag: flag whether job should spawn more jobs or not (Default value = True)

        Returns:

        """
        check_gradient_job = TrajectoryJob_check_gradient(data_id=data_object.get_id(),
                                                          parameters=parameters,
                                                          continue_flag=continue_flag)
        self._enqueue_job(check_gradient_job)

    def add_check_minima_jobs(self, data_object, run_object, continue_flag):
        """Adds a check_minima/optimize job to the queue.

        Args:
          data_object: data object for the job
          continue_flag: flag whether job should spawn more jobs or not
          run_object:  network_model required for the train job

        Returns:

        """
        if not self.do_check_minima:
            return
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
            new_data_object = self.data_container.get_data(current_id)
            new_data_object.steps[:] = [data_object.steps[minima_index]]
            new_data_object.parameters[:] = [data_object.parameters[minima_index]]
            new_data_object.losses[:] = [data_object.losses[minima_index]]
            new_data_object.gradients[:] = [data_object.gradients[minima_index]]
            self.data_container.update_data(new_data_object)
            self.add_train_job(
                data_object=new_data_object,
                run_object=run_object,
                continue_flag=continue_flag)

    def add_extract_minima_job(self, data_object, parameters, continue_flag):
        """Adds an extract job to the queue.

        Args:
          data_object: id associated with data object for the job
          parameters: parameters for analysis
          continue_flag: flag whether job should spawn more jobs or not

        Returns:

        """
        if not self.do_check_minima:
            return
        extract_job = TrajectoryJob_extract_minimium_candidates(
            data_id=data_object.get_id(),
            parameters=parameters,
            continue_flag=continue_flag)
        self._enqueue_job(extract_job)

    def add_prune_job(self, data_object, network_model, continue_flag):
        """Adds a prune job to the queue.

        Args:
          data_object: data object for the job
          network_model: neural network model
          continue_flag: flag whether job should spawn more jobs or not

        Returns:

        """
        prune_job = TrajectoryJob_prune(data_id=data_object.get_id(),
                                        network_model=network_model,
                                        continue_flag=False)
        self._enqueue_job(prune_job)

    def add_sample_job(self, data_object, run_object, continue_flag=False):
        """Adds a run job to the queue.

        Args:
          data_object: data object for the job
          run_object: neural network object for running the graph
          continue_flag: flag whether job should spawn more jobs or not (Default value = False)

        Returns:

        """
        data_object = self.instantiate_data_object(data_object)
        if len(data_object.legs_at_step) > 0:
            initial_step = data_object.legs_at_step[-1]
        else:
            initial_step = 0
        # TODO: parameters needs to be properly adapted to ensemble of walkers case
        if len(data_object.parameters) > 0:
            parameters = [data_object.parameters[-1]]
        else:
            parameters = None
        sample_job = TrajectoryJob_sample(data_id=data_object.get_id(),
                                          network_model=run_object,
                                          initial_step=initial_step,
                                          parameters=parameters,
                                          continue_flag=continue_flag)
        self._enqueue_job(sample_job)

    def add_train_job(self, data_object, run_object, continue_flag=False):
        """Adds a run job to the queue.

        Args:
          data_object: data object for the job
          run_object: neural network object for running the graph
          continue_flag: flag whether job should spawn more jobs or not (Default value = False)

        Returns:

        """
        data_object = self.instantiate_data_object(data_object)
        # TODO: parameters needs to be properly adapted to ensemble of walkers case
        train_job = TrajectoryJob_train(data_id=data_object.get_id(),
                                        network_model=run_object,
                                        initial_step=data_object.steps[-1],
                                        parameters=[data_object.parameters[-1]],
                                        continue_flag=continue_flag)
        self._enqueue_job(train_job)

    def trajectory_ended(self, data_id):
        """Provides a hook for derived classes to do something when a trajectory
        is terminated.

        Args:
          data_id: data id to this trajectory

        Returns:

        """
        pass

    def leg_ended(self, data_id):
        """Provides a hook for derived classes to do something when a leg
        is terminated.

        Args:
          data_id: data id to this trajectory

        Returns:

        """
        pass

    def run_next_job(self, run_object, analyze_object):
        """Takes the next job from the start of the queue and runs it.
        Will add new jobs to queue depending on the result of the run job.

        Args:
          run_object: neural network object for run
          analyze_object: parameter object for analysis

        Returns:

        """
        usable_job = False
        while not usable_job:
            if self.number_processes == 0:
                current_job = self.queue.popleft()
            else:
                current_job = self.queue.get()
            data_id = current_job.get_data_id()

            # check whether data_id is in list
            usable_job = self.used_data_ids.count(data_id) == 0
            logging.info("Job #"+str(current_job.get_job_id())+" is " \
                         +("NOT " if not usable_job else "")+"usable.")

            if not usable_job:
                # decrease JoinableQueue counter for this, as we need to re-add
                if self.number_processes != 0:
                    self.queue.task_done()
                self._enqueue_job(current_job)

        # append data id to list to mark it as in use
        self.used_data_ids.append(data_id)

        logging.info("Current job #"+str(current_job.get_job_id())+": " \
                     +current_job.job_type+" on data #"+str(data_id))

        data_object = self.data_container.get_data(data_id)
        logging.debug("Old data is "+str(data_object))

        data_object, continue_flag = current_job.run(data_object)

        self.data_container.update_data(data_object)

        logging.debug("New data is "+str(data_object))

        # remove id from list again
        self.used_data_ids.remove(data_id)

        if current_job.job_type in ["sample", "train"]:
            self.leg_ended(data_id)

        logging.info("Continue? "+str(continue_flag))
        if continue_flag:
            if len(data_object.legs_at_step) >= self.max_legs and \
                            current_job.job_type in ["analyze", "check_gradient"]:
                logging.info("Maximum number of legs exceeded, not adding any more jobs of type " \
                             +current_job.job_type+" for data id "+str(data_object.get_id())+".")
                self.trajectory_ended(data_id)
                if current_job.job_type == "analyze":
                    self.add_extract_minima_job(data_object, analyze_object, False)
                    logging.info("Added extract minima job")
            else:
                if current_job.job_type == "sample":
                    for i in range(self.number_pruning):
                        self.add_prune_job(data_object, run_object, False)
                    logging.info("Added prune job(s)")
                    self.add_analyze_job(data_object, analyze_object, current_job.continue_flag)
                    logging.info("Added analyze job")

                elif current_job.job_type == "train":
                    self.add_check_gradient_job(data_object)
                    logging.info("Added check gradient job")

                elif current_job.job_type == "check_gradient":
                    self.add_train_job(data_object=data_object,
                                       run_object=run_object,
                                       continue_flag=current_job.continue_flag)
                    logging.info("Added train job")

                elif current_job.job_type == "extract_minimium_candidates":
                    self.add_check_minima_jobs(data_object, run_object, True)
                    logging.info("Added check minima jobs")

                elif current_job.job_type == "analyze":
                    self.add_sample_job(data_object=data_object,
                                        run_object=run_object,
                                        continue_flag=current_job.continue_flag)
                    logging.info("Added sample job")

                else:
                    logging.warning("Unknown job type "+current_job.job_type+"!")
        else:
            self.trajectory_ended(data_id)
            if current_job.job_type == "analyze":
                self.add_extract_minima_job(data_object, analyze_object, False)
                logging.info("Added extract minima job")
            else:
                logging.info("Not adding.")

