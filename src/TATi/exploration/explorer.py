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
        data_container = self.queue.get_data_container()
        for i in range(len(idx_corner)):
            logging.debug("Current corner point is (first and last five shown):" \
                          +str(parameters[idx_corner[i]][:5])+" ... "+str(parameters[idx_corner[i]][-5:]))
            current_id = data_container.add_empty_data(type="sample")
            data_object = data_container.get_data(current_id)
            data_object.steps[:] = [steps[idx_corner[i]]]
            data_object.parameters[:] = [parameters[idx_corner[i]]]
            data_object.losses[:] = [losses[idx_corner[i]]]
            data_object.gradients[:] = [1]
            data_container.update_data(data_object)

            self.queue.add_sample_job(
                data_object=data_object,
                run_object=network_model,
                continue_flag=True)
            cornerpoints.append( [data_object.steps[-1], data_object.losses[-1], data_object.parameters[-1]] )
        return cornerpoints

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

    def get_corner_points(self, trajectory, losses, parameters,
                          number_of_corner_points):
        """ Returns the corner points for a given

        :param trajectory: trajectory as parameters (i.e. weights and biases) per step
        :param losses: loss per step
        :param parameters: parameter struct controlling the diffusion map analysis
        :param number_of_corner_points: number of corner points to return
        :return: list of indices of the corner points with respect to the given trajectory
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
            dmap_kernel = np.zeros( (np.shape(trajectory)[0], np.shape(trajectory)[0]) )
            # override landmarks to skip computation
            parameters.landmarks = 0
        logging.info("Global diffusion map eigenvalues: "+str(dmap_eigenvalues))

        # c. find number of points maximally apart
        idx_corner = self.find_corner_points(
                dmap_eigenvectors, number_of_corner_points)
        return idx_corner

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