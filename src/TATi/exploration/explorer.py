from TATi.exploration.trajectoryjobqueue import TrajectoryJobQueue
from TATi.exploration.trajectoryprocessqueue import TrajectoryProcessQueue
from TATi.TrajectoryAnalyser import compute_diffusion_maps

import logging

import numpy as np

import scipy.sparse
import scipy.spatial.distance as scidist


class Explorer(object):
    """ Explorer is the Python API class for performing exploration of the loss
    manifold of a neural network.

    """

    def __init__(self, parameters, max_legs=20, use_processes=0, number_pruning=0, manager=None):
        """ Initializes the explorer class with its internal instances.

        :param parameters: parameter struct for steering exploration
        :param max_legs: maximum number of legs per trajectory
        :param use_processes: whether to use a single or multiple processes (using
                multiple processes allows to explore in parallel but needs to build
                multiple copies of the graph, one per process)
        :param number_pruning: number of pruning jobs added after trajectory has ended
        :param manager: multiprocessing's manager object to control shared instances
                in case of multiple processes
        """
        self.use_processes = use_processes != 0
        self.parameters = parameters
        if use_processes == 0:
            self.queue = TrajectoryJobQueue(max_legs, number_pruning)
        else:
            self.queue = TrajectoryProcessQueue(parameters, number_pruning, number_processes=use_processes, manager=manager)
        self.max_distance = 0. # helps to estimate when exploration has finished

    def add_used_data_ids_list(self, _list):
        """ Pass function through to TrajectoryQueue

        :param _list: list to use for storing currently used data ids
        """
        self.queue.add_used_data_ids_list(_list)

    def spawn_starting_trajectory(self, network_model, number_trajectories=3):
        """ Begin exploration by sampling an initial starting trajectory.

        :param network_model: model of neural network with Session for sample and optimize jobs
        """
        for i in range(1,number_trajectories+1):
            self.queue.add_sample_job(
                data_object=None,
                run_object=network_model,
                continue_flag=True)

    def spawn_corner_trajectories(self, steps, parameters, losses, idx_corner, network_model):
        """ Run further trajectories for a given list of corner points.

        :param steps: continuous step number per step
        :param parameters: trajectory as parameters (i.e. weights and biases) per step
        :param losses: loss per step
        :param idx_corner: list of indices of all corner points w.r.t. trajectory
        :param network_model: model of neural network with Session for sample and optimize jobs
        :return: added cornerpoints as array
        """
        # d. spawn new trajectories from these points
        cornerpoints = []
        for i in range(len(idx_corner)):
            self.spawn_new_trajectory(steps[idx_corner[i]], parameters[idx_corner[i]],
                                      losses[idx_corner[i]], 1, network_model)
            cornerpoints.append( [steps[idx_corner[i]], losses[idx_corner[i]], parameters[idx_corner[i]]] )
        return cornerpoints

    def spawn_new_trajectory(self, step, parameters, loss, gradient, network_model):
        """ Run further trajectories for a given list of corner points.

        :param steps: continuous step number per step
        :param parameters: trajectory as parameters (i.e. weights and biases) per step
        :param losses: loss per step
        :param idx_corner: list of indices of all corner points w.r.t. trajectory
        :param network_model: model of neural network with Session for sample and optimize jobs
        :return: added cornerpoints as array
        """
        # d. spawn new trajectories from these points
        data_container = self.queue.get_data_container()
        logging.debug("Starting new trajectory from (first and last five shown):" \
                      +str(parameters[:5])+" ... "+str(parameters[-5:]))
        current_id = data_container.add_empty_data(type="sample")
        data_object = data_container.get_data(current_id)
        data_object.steps[:] = [step]
        data_object.parameters[:] = [parameters]
        data_object.losses[:] = [loss]
        data_object.gradients[:] = [gradient]
        data_container.update_data(data_object)

        self.queue.add_sample_job(
            data_object=data_object,
            run_object=network_model,
            continue_flag=True)

    def combine_sampled_trajectories(self):
        """ Combines all trajectories contained in the internal container.

        :return: combined parameters and losses for diffusion map analysis
        """
        steps = []
        parameters = []
        losses = []
        data_container = self.queue.get_data_container()
        for id in data_container.get_ids():
            data_object = data_container.get_data(id)
            if data_object.type == "sample":
                steps.extend( data_object.steps )
                parameters.extend( data_object.parameters )
                losses.extend( data_object.losses )
        return steps, parameters, losses

    @staticmethod
    def find_corner_points(dmap_eigenvectors, number_corner_points):
        """ Finds corner points given the diffusion map eigenvectors of a trajectory.

        :param dmap_eigenvectors: diffusion map eigenvector matrix
        :param number_corner_points: desired number of corner points
        :return: indices of the corner points with respect to trajectory
        """
        if number_corner_points == 0:
            return []

        # select a random point and compute distances to it
        select_first = "dominant_eigenmode"  # "random"
        if select_first == "random":
            m = np.shape(dmap_eigenvectors)[0]
            idx_corner = np.random.randint(m)

            dist = scidist.cdist(dmap_eigenvectors[[idx_corner], :], dmap_eigenvectors)[0]
            idx_corner = [np.argmax(dist)]

        elif select_first == "dominant_eigenmode":
            # find first cornerstone as maximum on dominant eigenvector
            idx_corner = [np.argmax(dmap_eigenvectors[:, 0])]
        else:
            assert (False)

        logging.debug('idx_corner ')
        logging.debug(idx_corner)
        # iteration to find the other cornerstones
        for k in np.arange(1, number_corner_points):
            # update minimum distance to existing cornerstones
            if (k > 1):
                dist = np.minimum(dist, scidist.cdist(dmap_eigenvectors[[idx_corner[-1]], :], dmap_eigenvectors)[0])
            else:
                dist = scidist.cdist(dmap_eigenvectors[idx_corner, :], dmap_eigenvectors)[0]
            # select new cornerstone
            idx_corner.append(np.argmax(dist))

        return idx_corner

    def perform_diffusion_map_analysis(self, trajectory, losses, parameters):
        """ Returns the eigenvectors and eigenvalues of the diffusion map

        :param trajectory: trajectory as parameters (i.e. weights and biases) per step
        :param losses: loss per step
        :param parameters: parameter struct controlling the diffusion map analysis
        :return: eigenvectors, eigenvalues
        """
        try:
            dmap_eigenvectors, dmap_eigenvalues, dmap_kernel = compute_diffusion_maps( \
                traj=trajectory, \
                beta=parameters.inverse_temperature, \
                loss=losses, \
                nrOfFirstEigenVectors=parameters.number_of_eigenvalues, \
                method=parameters.diffusion_map_method,
                use_reweighting=parameters.use_reweighting)
        except scipy.sparse.linalg.eigen.arpack.ArpackNoConvergence:
            logging.error(": Vectors were non-convergent.")
            dmap_eigenvectors = np.zeros( (np.shape(trajectory)[0], parameters.number_of_eigenvalues) )
            dmap_eigenvalues = np.zeros( (parameters.number_of_eigenvalues) )
            #dmap_kernel = np.zeros( (np.shape(trajectory)[0], np.shape(trajectory)[0]) )

        return dmap_eigenvectors, dmap_eigenvalues

    def run_all_jobs(self, network_model, parameters):
        """ Run all jobs currently found in the TrajectoryJob queue.

        :param network_model: model of neural network with Session for sample and optimize jobs
        :param parameters: parameter struct for analysis jobs
        """
        self.queue.run_all_jobs(network_model, parameters)

    def get_run_info_and_trajectory(self):
        """ This combines all stored run_info and trajectory and returns them
        as a single array.

        This returns the same struct's as does thermodynamicanalyticstoolkit.model.model's
        sample() and train() functions.

        :return: run_info and trajectory
        """
        data_container = self.queue.get_data_container()

        run_info = []
        for current_id in data_container.data.keys():
            run_lines_per_leg = data_container.data[current_id].run_lines
            for leg_nr in range(len(run_lines_per_leg)):
                run_lines = run_lines_per_leg[leg_nr]
                for row in range(len(run_lines.index)):
                    run_line = run_lines.iloc[row,:]
                    run_line[0] = current_id
                    run_info.append(run_line)

        trajectory = []
        for current_id in data_container.data.keys():
            trajectory_lines_per_leg = data_container.data[current_id].trajectory_lines
            for leg_nr in range(len(trajectory_lines_per_leg)):
                trajectory_lines = trajectory_lines_per_leg[leg_nr]
                for row in range(len(trajectory_lines.index)):
                    trajectory_line = trajectory_lines.iloc[row,:]
                    trajectory_line[0] = current_id
                    trajectory.append(trajectory_line)

        return run_info, trajectory

    def _get_largest_separation_distance(self, dmap_eigenvectors):
        """ Helper function to get the largest separation distance over all
        eigenvectors

        :param dmap_eigenvectors: eigenvectors to look for changes
        :return: largest distance, i.e. maximum (max-min) value over all
        """
        distances = []
        for i in range(np.shape(dmap_eigenvectors)[1]):
            distances.append(np.argmax(dmap_eigenvectors[:, i]) - np.argmin(dmap_eigenvectors[:, i]))
        return max(distances)

    def has_exploration_finished(self, dmap_eigenvectors):
        """ Checks whether the exploration in the current basin has finished.

        We check this by looking at the diffusion map's eigenvector.
        If the maximally separate distances per eigenvector has not
        changed, then we have not made any progress and seem to have exhausted
        the current basin.

        :param dmap_eigenvectors: eigenvectors to look for changes
        :return: true - has finished, false - exploration needs to continue
        """
        if dmap_eigenvectors is None:
            return True
        else:
            new_distance = self._get_largest_separation_distance(dmap_eigenvectors)
            logging.debug("New distance is "+str(new_distance)+", compared to old distance "
                          +str(self.max_distance))
            if new_distance > self.max_distance:
                self.max_distance = new_distance
                return False
            else:
                return True

    def explore_basin(self, network_model):
        """ This explores the current basin in the loss manifold fully at the
        currently set temperature.

        :param network_model: network_model used for sampling
        :return: eigenvectors, eigenvalues of diffusion kernel of the sampled trajectory
        """
        dmap_eigenvectors = None
        cornerpoints = []
        self.max_distance = 0.
        while True:
            # a. combine all trajectories
            steps, parameters, losses = self.combine_sampled_trajectories()

            # b. perform diffusion map analysis
            dmap_eigenvectors, dmap_eigenvalues = self.perform_diffusion_map_analysis(
                parameters, losses, self.parameters)
            if dmap_eigenvalues[0] == 0.:
                # override landmarks to skip computation
                parameters.landmarks = 0
            logging.info("Global diffusion map eigenvalues: " + str(dmap_eigenvalues))

            # check for exhausted exploration
            if self.has_exploration_finished(dmap_eigenvectors):
                break

            # c. get corner points
            idx_corner = self.find_corner_points(dmap_eigenvectors,
                                                self.parameters.number_of_parallel_trajectories)

            # d. spawn new trajectories from these points
            cornerpoints.append(
                self.spawn_corner_trajectories(steps, parameters, losses, idx_corner,
                                               network_model))

            # e. run all trajectories till terminated
            self.run_all_jobs(network_model, self.parameters)

        return steps, parameters, losses, dmap_eigenvectors, dmap_eigenvalues