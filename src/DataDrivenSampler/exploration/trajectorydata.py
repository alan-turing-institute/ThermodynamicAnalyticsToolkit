import logging

class TrajectoryData(object):
    ''' This class contains all data associated with a single trajectory.
    Trajectory steps are bundled into consecutive legs. Over these legs
    we check diffusion map values for convergence and end trajectory if this
    is the case.

    This is:
    -# id associated with this trajectory
    -# parameters per step
    -# loss, gradients per step
    -# eigenvectors and eigenvalues of diffusion map analysis per leg.

    '''

    def __init__(self, _id, _type = "sample"):
        ''' Initialize data object with a valid id

        :param _id: id of the data object
        '''
        self.id = _id

        # these are per step
        self.steps = []
        self.parameters = []
        self.losses = []
        self.gradients = []
        self.averages_lines = []
        self.run_lines = []
        self.trajectory_lines = []

        # these are per leg
        self.legs_at_step = []  # this gives the real-world trajectory step at each leg
        self.index_at_leg = []  # this gives the offset to parameters, ... at each leg
        self.diffmap_eigenvectors = []
        self.diffmap_eigenvalues = []

        # candidates for minima
        self.minimum_candidates = []
        self.hessian_eigenvalues = []

        # indicates whether this trajectory is done
        self.is_pruned = False

        # type of dynamics that created the trajectory in this data object
        self.type = _type

        # model filename used by trajectoryprocess'es
        self.model_filename = None

    def get_id(self):
        """ Return the unique id of this data object

        :return: unique id of object
        """
        return self.id

    def add_run_step(self, _steps, _parameters, _losses, _gradients,
                     _averages_lines=None, _run_lines=None, _trajectory_lines=None):
        """ Appends all values from a single run (one leg) to the specific
        internal containers for later analysis

        :param _steps: step per array component
        :param _parameters: (weight and bias) parameters of neural network as flattened vector
        :param _losses: loss/potential energy
        :param _gradients: gradient norm per step
        :param _averages_lines: single pandas dataframe of averages line per step
        :param _run_lines: single pandas dataframe of run info line per step
        :param _trajectory_lines: single pandas dataframe of trajectory line per step
        """
        self.steps.extend(_steps)
        self.index_at_leg.append( len(self.parameters) )
        self.parameters.extend(_parameters)
        self.losses.extend(_losses)
        self.gradients.extend(_gradients)
        # trajectories need to append continuously w.r.t steps
        if (len(self.legs_at_step) > 0):
            logging.debug("Last leg ended at "+str(self.legs_at_step[-1])+", next starts at "+str(_steps[0]))
            assert( (len(self.legs_at_step) == 0) or (self.legs_at_step[-1] < _steps[0]) )
        self.legs_at_step.append(_steps[-1])
        if _averages_lines is not None:
            self.averages_lines.append(_averages_lines)
        if _run_lines is not None:
            self.run_lines.append(_run_lines)
        if _trajectory_lines is not None:
            self.trajectory_lines.append(_trajectory_lines)
        assert( self.check_size_consistency() )

    def check_size_consistency(self):
        """ Checks whether the sizes of all the arrays are consistent.

        :return: True - sizes match, False - something is broken
        """
        status = True
        status = status and ( len(self.parameters) == len(self.losses) )
        status = status and ( len(self.parameters) == len(self.gradients) )
        status = status and ( len(self.legs_at_step) == len(self.index_at_leg) )
        status = status and ( (len(self.legs_at_step) == len(self.averages_lines)) \
                or (len(self.averages_lines)== 0))
        status = status and ( (len(self.legs_at_step) == len(self.run_lines)) \
                or (len(self.run_lines)== 0))
        status = status and ( (len(self.legs_at_step) == len(self.trajectory_lines) ) \
                or (len(self.trajectory_lines) == 0))
        status = status and ( len(self.diffmap_eigenvalues) == len(self.diffmap_eigenvectors) )
        return status

    def add_analyze_step(self, _eigenvectors, _eigenvalues):
        """ Adds diffusion map analysis values per leg to specific containers.

        :param _eigenvectors: first dominant eigenvectors of diffusion map
        :param _eigenvalues:  first dominant eigenvalues of diffusion map
        """
        self.diffmap_eigenvectors.append(_eigenvectors)
        self.diffmap_eigenvalues.append(_eigenvalues)
        assert( self.check_size_consistency() )
