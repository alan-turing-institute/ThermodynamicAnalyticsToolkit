""" @package docstring
CommandlineOptions is a specialization of Options for options parsed from the
command-line.
"""

import argparse
import logging
import sys
from TATi.common import get_filename_from_fullpath, get_list_from_string
from TATi.options.pythonoptions import PythonOptions
from TATi.version import get_package_version, get_build_hash

def str2bool(v):
    # this is the answer from stackoverflow https://stackoverflow.com/a/43357954/1967646
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def react_generally_to_options(FLAGS, unparsed):
    """ Extracted behavior for options shared between sampler and optimizer
    here for convenience.

    :param FLAGS: parsed cmd-line options as produced by argparse.parse_known_args()
    :param unparsed: unparsed cmd-line options as produced by argparse.parse_known_args()
    """
    if FLAGS.version:
        # give version and exit
        print(get_filename_from_fullpath(
            sys.argv[0]) + " " + get_package_version() + " -- version " + get_build_hash())
        sys.exit(0)

    # setup log level
    if FLAGS.verbose == None:
        FLAGS.verbose = 0
    # sys.stdout.write('Logging is at '+str(verbosity[arguments.verbose])+' level\n')

    logging.info("Using parameters: " + str(FLAGS))

    if len(unparsed) != 0:
        logging.error("There are unparsed parameters '" + str(unparsed) + "', have you misspelled some?")
        sys.exit(255)


def get_argparse_option_name(parser, *args, **kwargs):
    """ This is taken from `argparse.add_argument()` to extract the option name.

    :param parser: argparse instance
    :param args: arguments to `argparse.ArgumentParser.add_argument()`
    :param kwargs: keyword arguments to `argparse.ArgumentParser.add_argument()`
    :return: string name of option
    """
    if not args or len(args) == 1 and args[0][0] not in parser.prefix_chars:
        pos_args = parser._get_positional_kwargs(*args, **kwargs)
    else:
        pos_args = parser._get_optional_kwargs(*args, **kwargs)
    # print(pos_args)
    return pos_args['dest'].replace('--', '')


class CommandlineOptions(PythonOptions):
    """ CommandLineOptions extends the Options class to parse command-line
    options into the internal map.

    """
    def __init__(self):
        """ Creates the internal parser.
        """
        super(CommandlineOptions, self).__init__(add_keys=False)
        self._excluded_keys.append("parser")
        self.parser = argparse.ArgumentParser()

        # option keys whose values need to be converted before going into `_options_map`
        self._excluded_keys.append("_special_keys")
        self._special_keys = {}

    def _add_option_cmd(self, *args, **kwargs):
        """ Adds a purely cmd-line option to the internal parser and the options map.

        :param args: arguments to `argparse.ArgumentParser.add_argument()`
        :param kwargs: keyword arguments to `argparse.ArgumentParser.add_argument()`
        """
        option_name = get_argparse_option_name(self.parser, *args, **kwargs)
        self.add(option_name)
        self.parser.add_argument(*args, **kwargs)

    def _add_option(self, *args, **kwargs):
        """ Adds an option to the internal parser and the options map.

        Raises:
            KeyError (when default on `PythonOptions` has not been given.

        :param args: arguments to `argparse.ArgumentParser.add_argument()`
        :param kwargs: keyword arguments to `argparse.ArgumentParser.add_argument()`
        """
        option_name = get_argparse_option_name(self.parser, *args, **kwargs)
        self.add(option_name)
        kwargs.update({ "default": self._default_map[option_name]})
        if option_name in self._type_map.keys():
            kwargs.update({ "type": self._type_map[option_name]})
        elif option_name in self._list_type_map.keys():
            kwargs.update({ "type": []})
        else:
            assert(0)
        kwargs.update({ "help": self._description_map[option_name]})
        try:
            self.parser.add_argument(*args, **kwargs)
        except TypeError:
            del kwargs["type"]
            self.parser.add_argument(*args, **kwargs)

    def add_data_options_to_parser(self):
        """ Adding options common to both sampler and optimizer to argparse
        object for specifying the data set.
        """
        # please adhere to alphabetical ordering
        self._add_option('--batch_data_files', type=str, nargs='+',
                          help='Names of files to parse input data from')
        self._add_option('--batch_data_file_type', type=str, default="csv",
                          help='Type of the input files: csv (default), tfrecord')
        self._add_option('--input_dimension', type=int, default=2,
                          help='Number of input nodes, i.e. dimensionality of provided dataset')
        self._add_option('--output_dimension', type=int, default=1,
                          help='Number of output nodes, e.g. classes in a classification problem')

    def add_prior_options_to_parser(self):
        """ Adding options for setting up the prior enforcing to argparse
        """
        self._add_option('--prior_factor', type=float, default=None,
                          help='Enforce a prior by constraining parameters, this scales the wall repelling force')
        self._add_option('--prior_lower_boundary', type=float, default=None,
                          help='Enforce a prior by constraining parameters from below with a linear force')
        self._add_option('--prior_power', type=float, default=None,
                          help='Enforce a prior by constraining parameters, this sets the power of the wall repelling force')
        self._add_option('--prior_upper_boundary', type=float, default=None,
                          help='Enforce a prior by constraining parameters from above with a linear force')

    def add_model_options_to_parser(self):
        """ Adding options common to both sampler and optimizer to argparse
        object for specifying the model.
        """
        # please adhere to alphabetical ordering
        self._add_option('--batch_size', type=int, default=None,
                          help='The number of samples used to divide sample set into batches in one training step.')
        self._add_option('--do_hessians', type=str2bool, default=False,
                          help='Whether to add hessian computation nodes to graph, when present used by optimizer and explorer')
        self._add_option('--dropout', type=float, default=None,
                          help='Keep probability for training dropout, e.g. 0.9')
        self._add_option('--fix_parameters', type=str, default=None,
                          help='Fix parameters for sampling/training by stating "name=value;..."')
        self._add_option('--hidden_activation', type=str, default="relu",
                          help='Activation function to use for hidden layer: tanh, relu, linear')
        self._add_option('--hidden_dimension', type=str, nargs='+', default=[],
                          help='Dimension of each hidden layer, e.g. 8 8 for two hidden layers each with 8 nodes fully connected')
        self._add_option('--in_memory_pipeline', type=str2bool, default=True,
                          help='Whether to use an in-memory input pipeline (for small datasets) or the tf.Dataset module.')
        self._add_option('--input_columns', type=str, nargs='+', default="",
                          help='Pick a list of the following: x1, x1^2, sin(x1), cos(x1) or combinations with x1 representing the input dimension, e.g. x1 x2^2 sin(x3).')
        self._add_option('--loss', type=str, default="mean_squared",
                          help='Set the loss to be measured during sampling, e.g. mean_squared, log_loss, ...')
        self._add_option('--output_activation', type=str, default="tanh",
                          help='Activation function to use for output layer: tanh, relu, linear')
        self._add_option('--parse_parameters_file', type=str, default=None,
                          help='File to parse initial set of parameters from')
        self._add_option('--parse_steps', type=int, nargs='+', default=[],
                          help='Step(s) to parse from parse_parameters_file assuming multiple are present')
        self._add_option('--seed', type=int, default=None,
                          help='Seed to use for random number generators.')
        self._add_option('--summaries_path', type=str, default=None,
                          help='path to write TensorBoard summaries to')

    def add_common_options_to_parser(self):
        """ Adding options common to both sampler and optimizer to argparse
        object for specifying files and how to write them.
        """
        # please adhere to alphabetical ordering
        self._add_option('--averages_file', type=str, default=None,
                          help='CSV file name to write ensemble averages information such as average kinetic, potential, virial.')
        self._add_option('--burn_in_steps', type=int, default=0,
                          help='Number of initial steps to discard for averages ("burn in")')
        self._add_option('--every_nth', type=int, default=1,
                          help='Store only every nth trajectory (and run) point to files, e.g. 10')
        self._add_option('--inter_ops_threads', type=int, default=1,
                          help='Sets the number of threads to split up ops in between. NOTE: This hurts reproducibility to some extent '
                                 'because of parallelism.')
        self._add_option('--intra_ops_threads', type=int, default=None,
                          help='Sets the number of threads to use within an op, i.e. Eigen threads for linear algebra routines.')
        self._add_option('--max_steps', type=int, default=1000,
                         help='Number of steps to run trainer.')
        self._add_option('--number_walkers', type=int, default=1,
                          help='Number of dependent walkers to run. This will activate ensemble preconditioning if larger than 1.')
        self._add_option('--progress', type=str2bool, default=False,
                          help='Display progress bar with estimate of remaining time.')
        self._add_option('--restore_model', type=str, default=None,
                          help='Restore model (weights and biases) from a file.')
        self._add_option('--run_file', type=str, default=None,
                          help='CSV run file name to runtime information such as output accuracy and loss values.')
        self._add_option('--save_model', type=str, default=None,
                          help='Save model (weights and biases) to a file for later restoring.')
        self._add_option('--sql_db', type=str, default=None,
                          help='Supply file for writing timing information to sqlite database')
        self._add_option('--trajectory_file', type=str, default=None,
                          help='CSV file name to output trajectories of sampling, i.e. weights and evaluated loss function.')
        self._add_option('--verbose', '-v', action='count', default=1,
                          help='Level of verbosity during compare')
        self._add_option_cmd('--version', '-V', action="store_true",
                          help='Gives version information')

    def add_train_options_to_parser(self):
        """ Adding options common to train to argparse.
        """
        # please adhere to alphabetical ordering
        self._add_option('--learning_rate', type=float, default=0.03,
                         help='step width \Delta t to use during training/optimizing, e.g. 0.01')
        self._add_option('--optimizer', type=str, default="GradientDescent",
                         help='Choose the optimizer to use for sampling: GradientDescent')

    def add_sampler_options_to_parser(self):
        """ Adding options common to sampler to argparse.
        """
        # please adhere to alphabetical ordering
        self._add_option('--collapse_after_steps', type=int, default=100,
                          help='Number of steps after which to regularly collapse all dependent walkers to restart from a single position '
                                 'again, maintaining harmonic approximation for ensemble preconditioning. 0 will never collapse.')
        self._add_option('--covariance_blending', type=float, default=0.,
                          help='Blending between unpreconditioned gradient (0.) and preconditioning through covariance matrix from other '
                                 'dependent walkers')
        self._add_option('--friction_constant', type=float, default=0.,
                          help='friction to scale the influence of momenta')
        self._add_option('--inverse_temperature', type=float, default=0.,
                          help='Inverse temperature that scales the gradients')
        self._add_option('--hamiltonian_dynamics_time', type=float, default=10,
                          help='Time (steps times step width) between HMC acceptance criterion evaluation')
        self._add_option('--sampler', type=str, default="GeometricLangevinAlgorithm_1stOrder",
                          help='Choose the sampler to use for sampling: ' \
                                 + 'GeometricLangevinAlgorithm_1stOrder, GeometricLangevinAlgorithm_2ndOrder,' \
                                 + 'StochasticGradientLangevinDynamics, ' \
                                 + 'BAOAB, ' \
                                 + 'HamiltonianMonteCarlo, ')
        self._add_option('--sigma', type=float, default=1.,
                          help='Scale of noise injected to momentum per step for CCaDL.')
        self._add_option('--sigmaA', type=float, default=1.,
                          help='Scale of noise in convex combination for CCaDL.')
        self._add_option('--step_width', type=float, default=0.03,
                          help='step width \Delta t to use during sampling, e.g. 0.01')

    def _cast_to_type(self, value, type_name):
        if value is None:
            return value
        if type_name == str:
            try:
                return str(value)
            except ValueError:
                return [str(i) for i in get_list_from_string(value)]
        elif type_name == int:
            try:
                return int(value)
            except ValueError:
                return [int(i) for i in get_list_from_string(value)]
        elif type_name == float:
            try:
                return float(value)
            except ValueError:
                return [float(i) for i in get_list_from_string(value)]
        elif type_name == bool:
            try:
                return bool(value)
            except ValueError:
                return [bool(i) for i in get_list_from_string(value)]
        else:
            assert (0)

    def set(self, key, values):
        """ Override `set()` to enforce expected type before.

        :param key: option name
        :param value: option value
        :return:
        """
        if key in self._type_map.keys():
            designated_type = self._type_map[key]
            super(CommandlineOptions, self).set(
                key,
                self._cast_to_type(values, designated_type))
        elif key in self._list_type_map.keys():
            designated_type = self._list_type_map[key]
            if isinstance(values, list):
                set_list = []
                for value in values:
                    cast_value = self._cast_to_type(value, designated_type)
                    if isinstance(cast_value, list):
                        set_list.extend(cast_value)
                    else:
                        set_list.append(cast_value)
                super(CommandlineOptions, self).set(key, set_list)
            else:
                super(CommandlineOptions, self).set(
                    key,
                    [self._cast_to_type(values, designated_type)])
        else:
            ## key added to command-line only
            super(CommandlineOptions, self).set(key, values)

    def parse(self):
        """ Parses the command-line options for all known parameters.

        :return: unparsed (because unknown) parameters
        """
        FLAGS, unparsed = self.parser.parse_known_args()

        # react to flags in general
        react_generally_to_options(FLAGS, unparsed)

        # store parsed values internally
        for key in self._option_map.keys():
            value = getattr(FLAGS, key)
            if key not in self._special_keys:
                self.set(key, value)
            else:
                # pipe through the function whose names is stored in the dict
                self.set(key, getattr(self, self.self._special_keys[key])(value))

        return unparsed

    def react_to_common_options(self):
        """ Extracted behavior for options shared between sampler and optimizer
        here for convenience.

        """
        if self.number_walkers < 1:
            logging.error("The number of walkers needs to be positive.")
            sys.exit(255)

    def react_to_sampler_options(self):
        """ Extracted behavior checking validity of sampler options here for convenience.

        """
        if self.sampler in ["StochasticGradientLangevinDynamics",
                             "GeometricLangevinAlgorithm_1stOrder",
                             "GeometricLangevinAlgorithm_2ndOrder",
                             "BAOAB",
                             "CovarianceControlledAdaptiveLangevinThermostat",
                             "HamiltonianMonteCarlo_1stOrder",
                             "HamiltonianMonteCarlo_2ndOrder"] \
                        and self.inverse_temperature == 0.:
            logging.error("You are using a sampler but have not set the inverse_temperature.")
            sys.exit(255)

        if self.sampler in ["GeometricLangevinAlgorithm_1stOrder",
                             "GeometricLangevinAlgorithm_2ndOrder",
                             "BAOAB",
                             "CovarianceControlledAdaptiveLangevinThermostat"] \
                and self.friction_constant == 0.:
            logging.error("You have not set the friction_constant for a sampler that requires it.")
            sys.exit(255)

        if self.covariance_blending < 0.:
            logging.error("The covariance blending needs to be non-negative.")
            sys.exit(255)

