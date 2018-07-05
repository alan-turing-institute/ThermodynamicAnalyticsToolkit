from TATi.options.options import Options

import logging

class PythonOptions(Options):
    """ All options that the python interface understands.
    
    Example:
        >>> from TATi.options.pythonoptions import PythonOptions
        >>> PythonOptions.help()
        averages_file:             CSV file name to write ensemble averages information such as average kinetic, potential, virial
        batch_data_file_type:      type of the files to read input from
         <remainder omitted>
        >>> PythonOptions.help("batch_data_files")
        Option name: batch_data_files
        Description: set of files to read input from
        Type       : <class 'list'>
        Default    : []
        >>> options = PythonOptions(
        ... batch_data_files=["test.csv"],
        ... batch_size=3,
        ... max_steps=1000,
        ... optimizer="GradientDescent",
        ... )
    """

    ### gives a description for each option
    _description_map = {
        "averages_file": "CSV file name to write ensemble averages information "+ \
                         "such as average kinetic, potential, virial",
        "batch_data_files": "set of files to read input from",
        "batch_data_file_type": "type of the files to read input from",
        "batch_size": "The number of samples used to divide sample set into "+ \
                      "batches in one sampleing step.",
        "burn_in_steps": "number of initial steps to drop when computing averages",
        "collapse_after_steps": "collapse all walkers into a single position "+ \
                                "again after this many steps",
        "covariance_blending": "mixing for preconditioning matrix to gradient "+ \
                               "update, identity matrix plus this times the covariance matrix "+
                               "obtained from the other walkers, 0 - will never collapse",
        "diffusion_map_method": "name of method to use for diffusion map "+ \
                                "analysis: vanilla, TMDap, pydiffmap",
        "do_hessians": "whether to add hessian evaluation nodes to graph, used "+ \
                       "by optimzier and explorer",
        "dropout": "Keep probability for sampleing dropout, e.g. 0.9",
        "every_nth": "Store only every nth trajectory (and run) point to files, "+ \
                     "e.g. 10",
        "fix_parameters": "string formatted as 'name=value;...' with name of "+ \
                          "parameter fixed to value",
        "friction_constant": "friction to scale the influence of momenta",
        "hamiltonian_dynamics_time": "time w.r.t step_width for HMC sampler "+ \
                                     "passing between checking acceptance "+ \
                                     "criterion",
        "hidden_activation": "Activation function to use for hidden layer: "+ \
                             "tanh, relu, linear",
        "hidden_dimension": "Dimension of each hidden layer, e.g. 8 8 for two "+ \
                            "hidden layers each with 8 nodes fully connected",
        "in_memory_pipeline": "whether to feed the dataset from file in-memory "+ \
                              "(True) or not (False)",
        "input_columns": "Pick a list of the following: "+ \
                         "(1) x1, (2) x2, (3) x1^2, (4) x2^2, (5) sin(x1), "+ \
                         "(6) sin(x2).",
        "input_dimension": "number of input nodes/number of features",
        "inter_ops_threads": "size of thread pool used for independent ops",
        "intra_ops_threads": "size of thread pool used for parallelizing an op",
        "inverse_temperature": "Inverse temperature that scales the gradients",
        "learning_rate": "learning rate used during optimization, see also "+ \
                         "`step_width`",
        "loss": "Set the loss to be measured during sampling, e.g. "+ \
                "mean_squared, log_loss, ...",
        "max_steps": "Number of steps to run sampleer.",
        "number_of_eigenvalues": "",
        "number_walkers": "number of dependent walkers to activate ensemble "+ \
                          "preconditioning",
        "optimizer": "Choose the optimizer to use for sampling: GradientDescent",
        "output_activation": "Activation function to use for output layer: "+ \
                             "tanh, relu, linear",
        "output_dimension": "number of output nodes/number of classes",
        "parse_parameters_file": "parse neural network parameters from this "+ \
                                 "file on network creation",
        "parse_steps": "indicates the row (in column 'step') of the "+ \
                       "parse_parameters_file to use",
        "prior_factor": "factor for scaling prior force",
        "prior_lower_boundary": "lower boundary for wall-repelling prior force",
        "prior_power": "power of distance used in calculating force",
        "prior_upper_boundary": "upper boundary for wall-repelling prior force",
        "progress": "activate progress bar to show remaining time",
        "restore_model": "Restore model (weights and biases) from a file.",
        "run_file": "CSV run file name to runtime information such as output "+ \
                    "accuracy and loss values.",
        "sampler": "Choose the sampler to use for sampling: "+ \
                   "BAOAB,"+ \
                   "CovarianceControlledAdaptiveLangevin, "+ \
                   "GeometricLangevinAlgorithm_1stOrder, "+ \
                   "GeometricLangevinAlgorithm_2ndOrder, "+ \
                   "HamiltonianMonteCarlo, "+ \
                   "StochasticGradientLangevinDynamics",
        "save_model": "Save model (weights and biases) to a file for later "+ \
                      "restoring.",
        "seed": "Seed to use for random number generators.",
        "sigma": "Scale of noise injected to momentum per step for CCaDL.",
        "sigmaA": "Scale of noise in convex combination for CCaDL.",
        "sql_db": "Filename of sqlite3 database file to write iteration "+ \
                  "information to",
        "step_width": "step width \Delta t to use, e.g. 0.01",
        "summaries_path": "path to write summaries (for TensorBoard) to",
        "trajectory_file": "CSV file name to output trajectories of sampling, "+ \
                           "i.e. weights and evaluated loss function.",
        "use_reweighting": "",
        "verbose": "how much (debugging) information to print",
    }

    ## sets a default value for each option
    _default_map = {
        "averages_file": None,
        "batch_data_files": [],
        "batch_data_file_type": "csv",
        "batch_size": None,
        "burn_in_steps": 0,
        "collapse_after_steps": 100,
        "covariance_blending": 0.,
        "diffusion_map_method": "vanilla",
        "do_hessians": False,
        "dropout": None,
        "every_nth": 1,
        "fix_parameters": None,
        "friction_constant": 0.,
        "hamiltonian_dynamics_time": 1.,
        "hidden_activation": "relu",
        "hidden_dimension": "",
        "in_memory_pipeline": True,
        "input_columns": [],
        "input_dimension": 2,
        "inter_ops_threads": 1,
        "intra_ops_threads": None,
        "inverse_temperature": 1.,
        "learning_rate": 0.03,
        "loss": "mean_squared",
        "max_steps": 1000,
        "number_of_eigenvalues": 4,
        "number_walkers": 1,
        "optimizer": "GradientDescent",
        "output_activation": "tanh",
        "output_dimension": 1,
        "parse_parameters_file": None,
        "parse_steps": [],
        "prior_factor": 1.,
        "prior_lower_boundary": None,
        "prior_power": 1.,
        "prior_upper_boundary": None,
        "progress": False,
        "restore_model": None,
        "run_file": None,
        "sampler": "GeometricLangevinAlgorithm_1stOrder",
        "save_model": None,
        "seed": None,
        "sigma": 1.,
        "sigmaA": 1.,
        "sql_db": None,
        "step_width": 0.1,
        "summaries_path": None,
        "trajectory_file": None,
        "use_reweighting": False,
        "verbose": 0,
    }

    ## sets the type of each option
    _type_map = {
        "averages_file": str,
        "batch_data_file_type": str,
        "batch_size": int,
        "burn_in_steps": int,
        "collapse_after_steps": int,
        "covariance_blending": float,
        "diffusion_map_method": str,
        "do_hessians": bool,
        "dropout": float,
        "every_nth": int,
        "fix_parameters": str,
        "friction_constant": float,
        "hamiltonian_dynamics_time": float,
        "hidden_activation": str,
        "in_memory_pipeline": bool,
        "input_dimension": int,
        "inter_ops_threads": int,
        "intra_ops_threads": int,
        "inverse_temperature": float,
        "learning_rate": float,
        "loss": str,
        "max_steps": int,
        "number_of_eigenvalues": int,
        "number_walkers": int,
        "optimizer": str,
        "output_activation": str,
        "output_dimension": int,
        "parse_parameters_file": str,
        "prior_factor": float,
        "prior_lower_boundary": float,
        "prior_power": float,
        "prior_upper_boundary": float,
        "progress": bool,
        "restore_model": str,
        "run_file": str,
        "sampler": str,
        "save_model": str,
        "seed": int,
        "sigma": float,
        "sigmaA": float,
        "sql_db": str,
        "step_width": float,
        "summaries_path": str,
        "trajectory_file": str,
        "use_reweighting": bool,
        "verbose": int,
    }

    # this denotes the type as of specific type
    _list_type_map = {
        "batch_data_files": str,
        "hidden_dimension": int,
        "input_columns": str,
        "parse_steps": int,
    }

    @staticmethod
    def help(key=None):
        """ Prints help for each option or all option names if key is None

        :param key: name of option or None for list of options
        """
        if key is None:
            max_length = 0
            for key in PythonOptions._description_map.keys():
                if len(key) > max_length:
                    max_length = len(key)
            for key in sorted(PythonOptions._description_map.keys()):
                print(str(key)+": "+" "*(max_length-len(key))+PythonOptions._description_map[key])
        elif key in PythonOptions._description_map.keys():
            print("Option name: "+str(key))
            print("Description: "+PythonOptions._description_map[key])
            if key in PythonOptions._type_map.keys():
                print("Type       : "+str(PythonOptions._type_map[key]))
            elif key in PythonOptions._list_type_map.keys():
                print("Type       : list of " + str(PythonOptions._list_type_map[key]))
            else:
                assert(0)
            print("Default    : "+str(PythonOptions._default_map[key]))

    def __init__(self, add_keys = True, value_dict = {}):
        """ Init function to set various default values.

        :param add_keys: controls whether all default values are added as options
        :param value_dict: keyword arguments for setting parameters different to default
        """
        # make sure all keys are described
        super(PythonOptions, self).__init__()
        self._excluded_keys.append("_description_map")
        self._excluded_keys.append("_default_map")
        self._excluded_keys.append("_type_map")
        self._excluded_keys.append("_list_type_map")
        for key in self._default_map.keys():
            if key not in  self._description_map.keys():
                logging.error("Option "+str(key)+" missing in _description_map")
        for key in self._description_map.keys():
            if key not in self._default_map.keys():
                logging.error("Option " + str(key) + " missing in _default_map")
        combined_type_keys = list(self._type_map.keys())+list(self._list_type_map.keys())
        for key in combined_type_keys:
            if key not in self._default_map.keys():
                logging.error("Option " + str(key) + " missing in _type_map")
        assert ( sorted(self._default_map.keys()) == sorted(self._description_map.keys()) )
        assert ( sorted(self._default_map.keys()) == sorted(combined_type_keys) )

        if add_keys:
            for key in self._default_map.keys():
                self.add(key)
                if key in value_dict.keys():
                    self.set(key, value_dict[key])
                else:
                    self.set(key, self._default_map[key])
