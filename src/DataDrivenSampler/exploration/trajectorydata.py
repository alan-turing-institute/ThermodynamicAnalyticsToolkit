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

    def __init__(self, _id):
        ''' Initialize data object with a valid id

        :param _id: id of the data object
        '''
        self.id = _id

        # these are per step
        self.steps = []
        self.parameters = []
        self.losses = []
        self.gradients = []
        self.run_lines = []
        self.trajectory_lines = []

        # these are per leg
        self.legs_at_step = []
        self.diffmap_eigenvectors = []
        self.diffmap_eigenvalues = []

    def get_id(self):
        """ Return the unique id of this data object

        :return: unique id of object
        """
        return self.id

    def add_run_step(self, _steps, _parameters, _losses, _gradients,
                     _run_lines=None, _trajectory_lines=None):
        """ Appends all values from a single run (one leg) to the specific
        internal containers for later analysis

        :param _steps: step per array component
        :param _parameters: (weight and bias) parameters of neural network as flattened vector
        :param _losses: loss/potential energy
        :param _gradients: gradient norm per step
        :param _run_lines: single pandas dataframe of run info line per step
        :param _trajectory_lines: single pandas dataframe of trajectory line per step
        """
        self.steps.extend(_steps)
        self.parameters.extend(_parameters)
        self.losses.extend(_losses)
        self.gradients.extend(_gradients)
        assert( len(self.parameters) == len(self.losses) )
        assert( len(self.parameters) == len(self.gradients) )
        # trajectories need to append continuously w.r.t steps
        if (len(self.legs_at_step) > 0):
            print("Last leg ended at "+str(self.legs_at_step[-1])+", next starts at "+str(_steps[0]))
            assert( (len(self.legs_at_step) == 0) or (self.legs_at_step[-1] < _steps[0]) )
        self.legs_at_step.append(_steps[-1])
        if _run_lines is not None:
            self.run_lines.append(_run_lines)
        assert (len(self.legs_at_step) == len(self.run_lines))
        if _trajectory_lines is not None:
            self.trajectory_lines.append(_trajectory_lines)
            assert( len(self.legs_at_step) == len(self.trajectory_lines) )

    def add_analyze_step(self, _eigenvectors, _eigenvalues):
        """ Adds diffusion map analysis values per leg to specific containers.

        :param _eigenvectors: first dominant eigenvectors of diffusion map
        :param _eigenvalues:  first dominant eigenvalues of diffusion map
        """
        self.diffmap_eigenvectors.append(_eigenvectors)
        self.diffmap_eigenvalues.append(_eigenvalues)
        assert( len(self.diffmap_eigenvalues) == len(self.diffmap_eigenvectors) )
