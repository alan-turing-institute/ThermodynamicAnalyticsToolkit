
class MockFlags:
    """ This class mimicks a parsed arguments structure returned from the module
    argparse.parse_known_arguments().
    """

    def __init__(self,
                 averages_file=None,
                 batch_data_files=[],
                 batch_data_file_type="csv",
                 batch_size=None,   # this does not help yet, python API does not check batch_size?
                 burn_in_steps=0,
                 collapse_after_steps=100,
                 covariance_blending=0.,
                 diffusion_map_method="vanilla",
                 do_hessians=False,
                 dropout=None,
                 every_nth=1,
                 fix_parameters=None,
                 friction_constant=0.,
                 hamiltonian_dynamics_time=10,
                 hidden_activation="relu",
                 hidden_dimension="",
                 in_memory_pipeline=True,
                 input_columns="",
                 input_dimension=2,
                 inter_ops_threads=1,
                 intra_ops_threads=None,
                 inverse_temperature=1.,
                 loss="mean_squared",
                 max_steps=1000,
                 number_of_eigenvalues=4,
                 number_walkers=1,
                 optimizer="GradientDescent",
                 output_activation="tanh",
                 output_dimension=1,
                 parse_parameters_file=None,
                 parse_steps=[],
                 prior_factor=1.,
                 prior_lower_boundary=None,
                 prior_power=1.,
                 prior_upper_boundary=None,
                 restore_model=None,
                 run_file=None,
                 sampler="GeometricLangevinAlgorithm_1stOrder",
                 save_model=None,
                 seed=None,
                 sigma=1.,
                 sigmaA=1.,
                 step_width=0.03,
                 trajectory_file=None,
                 use_reweighting=False,
                 verbose=0
    ):
        """ Init function to set various default values.

        :param averages_file: csv file name to write averages to
        :param batch_data_files: set of files to read input from
        :param batch_data_file_type: type of the files to read input from
        :param batch_size: The number of samples used to divide sample set into batches in one sampleing step.
        :param burn_in_steps: number of initial steps to drop when computing averages
        :param collapse_after_steps: collapse all walkers into a single position again after this many steps
        :param covariance_blending: mixing for preconditioning matrix to gradient
                update, identity matrix plus this times the covariance matrix obtained
                from the other walkers, 0 - will never collapse
        :param diffusion_map_method:
        :param do_hessians: whether to add hessian evaluation nodes to graph, used bz optimzier and explorer
        :param dropout: Keep probability for sampleing dropout, e.g. 0.9
        :param every_nth: Store only every nth trajectory (and run) point to files, e.g. 10
        :param fix_parameters: string formatted as "name=value;..." with name of parameter fixed to value
        :param friction_constant: friction to scale the influence of momenta
        :param hamiltonian_dynamics_time: number of steps in HMC sampler for checking acceptance criterion
        :param hidden_activation: Activation function to use for hidden layer: tanh, relu, linear
        :param hidden_dimension: Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected
        :param in_memory_pipeline: whether to feed the dataset from file in-memory (True) or not (False)
        :param input_columns: Pick a list of the following: (1) x1, (2) x2, (3) x1^2, (4) x2^2, (5) sin(x1), (6) sin(x2).
        :param input_dimension: number of input nodes/number of features
        :param inter_ops_threads: size of thread pool used for independent ops
        :param intra_ops_threads: size of thread pool used for parallelizing an op
        :param inverse_temperature: Inverse temperature that scales the gradients
        :param loss: Set the loss to be measured during sampling, e.g. mean_squared, log_loss, ...
        :param max_steps: Number of steps to run sampleer.
        :param number_of_eigenvalues:
        :param number_walkers: number of dependent walkers to activate ensemble preconditioning
        :param optimizer: Choose the optimizer to use for sampling: GradientDescent
        :param output_activation: Activation function to use for output layer: tanh, relu, linear
        :param output_dimension: number of output nodes/number of classes
        :param parse_parameters_file: parse neural network parameters from this file on network creation
        :param parse_steps: indicates the row (in column "step") of the parse_parameters_file to use
        :param prior_factor: factor for scaling prior force
        :param prior_lower_boundary: lower boundary for wall-repelling prior force
        :param prior_power: power of distance used in calculating force
        :param prior_upper_boundary: upper boundary for wall-repelling prior force
        :param restore_model: Restore model (weights and biases) from a file.
        :param run_file: CSV run file name to runtime information such as output accuracy and loss values.
        :param sampler: Choose the sampler to use for sampling: GeometricLangevinAlgorithm_1stOrder, GeometricLangevinAlgorithm_2ndOrder, StochasticGradientLangevinDynamics
        :param save_model: Save model (weights and biases) to a file for later restoring.
        :param seed: Seed to use for random number generators.
        :param sigma: Scale of noise injected to momentum per step for CCaDL.
        :param sigmaA: Scale of noise in convex combination for CCaDL.
        :param step_width: step width \Delta t to use, e.g. 0.01
        :param trajectory_file: CSV file name to output trajectories of sampling, i.e. weights and evaluated loss function.
        :param use_reweighting:
        :param verbose: how much (debugging) information to print
        """
        self.averages_file = averages_file
        self.batch_data_files = batch_data_files
        self.batch_data_file_type = batch_data_file_type
        self.batch_size = batch_size
        self.burn_in_steps = burn_in_steps
        self.collapse_after_steps = collapse_after_steps
        self.covariance_blending = covariance_blending
        self.diffusion_map_method = diffusion_map_method
        self.dimension = 0 # keeps the input dimension later
        self.do_hessians = do_hessians
        self.dropout = dropout
        self.every_nth = every_nth
        self.fix_parameters = fix_parameters
        self.friction_constant = friction_constant
        self.hamiltonian_dynamics_time = hamiltonian_dynamics_time
        self.hidden_activation = hidden_activation
        self.hidden_dimension = hidden_dimension
        self.in_memory_pipeline = in_memory_pipeline
        self.input_columns = input_columns
        self.input_dimension = input_dimension
        self.inter_ops_threads = inter_ops_threads
        self.intra_ops_threads = intra_ops_threads
        self.inverse_temperature = inverse_temperature
        self.loss = loss
        self.max_steps = max_steps
        self.number_of_eigenvalues = number_of_eigenvalues
        self.number_walkers = number_walkers
        self.optimizer = optimizer
        self.output_activation = output_activation
        self.output_dimension = output_dimension
        self.parse_parameters_file = parse_parameters_file
        self.parse_steps = parse_steps
        self.prior_factor = prior_factor
        self.prior_lower_boundary = prior_lower_boundary
        self.prior_power = prior_power
        self.prior_upper_boundary = prior_upper_boundary
        self.restore_model = restore_model
        self.run_file = run_file
        self.sampler = sampler
        self.save_model = save_model
        self.seed = seed
        self.sigma = sigma
        self.sigmaA = sigmaA
        self.step_width = step_width
        self.trajectory_file = trajectory_file
        self.use_reweighting = use_reweighting
        self.verbose = verbose
