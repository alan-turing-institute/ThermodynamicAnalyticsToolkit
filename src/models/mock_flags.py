from DataDrivenSampler.datasets.classificationdatasets import ClassificationDatasets


class MockFlags:
    """ This class mimicks a parsed arguments structure returned from the module
    argparse.parse_known_arguments().
    """

    def __init__(self,
                 batch_size=10,
                 data_type=ClassificationDatasets.SPIRAL,
                 dimension=10,
                 dropout=0.9,
                 every_nth=1,
                 friction_constant=0.,
                 hidden_activation="relu",
                 hidden_dimension="",
                 input_columns="1 2",
                 inverse_temperature=1.,
                 loss="mean_squared",
                 max_steps=1000,
                 noise=0.,
                 optimizer="GradientDescent",
                 output_activation="tanh",
                 restore_model=None,
                 run_file=None,
                 sampler="GeometricLangevinAlgorithm_1stOrder",
                 save_model=None,
                 seed=None,
                 step_width=0.03,
                 trajectory_file=None
    ):
        """ Init function to set various default values.

        :param batch_size: The number of samples used to divide sample set into batches in one sampleing step.
        :param data_type: Which data set to use: (0) two circles, (1) squares, (2) two clusters, (3) spiral.
        :param dimension: Number P of samples (Y^i,X^i)^P_{i=1} to generate for the desired dataset type.
        :param dropout: Keep probability for sampleing dropout, e.g. 0.9
        :param every_nth: Store only every nth trajectory (and run) point to files, e.g. 10
        :param friction_constant: friction to scale the influence of momenta
        :param hidden_activation: Activation function to use for hidden layer: tanh, relu, linear
        :param hidden_dimension: Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected
        :param input_columns: Pick a list of the following: (1) x1, (2) x2, (3) x1^2, (4) x2^2, (5) sin(x1), (6) sin(x2).
        :param inverse_temperature: Inverse temperature that scales the gradients
        :param loss: Set the loss to be measured during sampling, e.g. mean_squared, log_loss, ...
        :param max_steps: Number of steps to run sampleer.
        :param noise: Amount of noise in [0,1] to use.
        :param optimizer: Choose the optimizer to use for sampling: GradientDescent
        :param output_activation: Activation function to use for output layer: tanh, relu, linear
        :param restore_model: Restore model (weights and biases) from a file.
        :param run_file: CSV run file name to runtime information such as output accuracy and loss values.
        :param sampler: Choose the sampler to use for sampling: GeometricLangevinAlgorithm_1stOrder, GeometricLangevinAlgorithm_2ndOrder, StochasticGradientLangevinDynamics
        :param save_model: Save model (weights and biases) to a file for later restoring.
        :param seed: Seed to use for random number generators.
        :param step_width: step width \Delta t to use, e.g. 0.01
        :param trajectory_file: CSV file name to output trajectories of sampling, i.e. weights and evaluated loss function.
        """
        self.batch_size = batch_size
        self.data_type = data_type
        self.dimension = dimension
        self.dropout = dropout
        self.every_nth = every_nth
        self.friction_constant = friction_constant
        self.hidden_activation = hidden_activation
        self.hidden_dimension = hidden_dimension
        self.input_columns = input_columns
        self.inverse_temperature = inverse_temperature
        self.loss = loss
        self.max_steps = max_steps
        self.noise = noise
        self.optimizer = optimizer
        self.output_activation = output_activation
        self.restore_model = restore_model
        self.run_file = run_file
        self.sampler = sampler
        self.save_model = save_model
        self.seed = seed
        self.step_width = step_width
        self.trajectory_file = trajectory_file
