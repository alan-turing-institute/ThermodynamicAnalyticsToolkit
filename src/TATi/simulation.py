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

"""@package docstring
The simulation module contains the interface to generically access neural networks.

"""

import logging

import numpy as np

from TATi.model import Model
from TATi.models.evaluationcache import EvaluationCache
from TATi.models.parameters.networkparameter_adapter import NetworkParameterAdapter
from TATi.models.parameters.parameters import Parameters
from TATi.models.trajectories.trajectorydata import TrajectoryData
from TATi.options.pythonoptions import PythonOptions


class Simulation(object):
    """This class represents the Python interface to TATi that allows to
    access the neural network (including its loss function, parameters,
    ...) as a black-box function.
    
    The idea is that there is no need to worry about any of the neural
    network internals. An initial parameter structure is all that is
    needed and afterwards one may treat the whole thing as two (coupled)
    functions, namely the loss and the predictor, where the loss depends
    implicitly on the dataset and both use the set of parameters of the
    neural network.

    Args:

    Returns:

    """

    ### gives which parts of the interal state are affected by each option
    _affects_map = {
        "averages_file": ["network"],
        "batch_data_files": ["input", "network"],
        "batch_data_file_type": ["input", "network"],
        "batch_size": ["input"],
        "burn_in_steps": [],        # only used in python code
        "collapse_walkers": [], # only used in python code
        "covariance_after_steps": [], # only used in python code
        "covariance_blending": ["network"],
        "diffusion_map_method": [], # only used in python code
        "directions_file": [],
        "do_hessians": ["network"],
        "dropout": ["network"],
        "every_nth": [],            # only used in python code
        "fix_parameters": ["network"],
        "friction_constant": [],    # passed as placeholder
        "hamiltonian_dynamics_time": [], # used inly in python code, passed as placeholder next_eval_step_t
        "hidden_activation": ["network"],
        "hidden_dimension": ["network"],
        "in_memory_pipeline": ["input"],
        "input_columns": ["input"],
        "input_dimension": ["input", "network"],
        "inter_ops_threads": ["network"],    # affects Session instantiation
        "intra_ops_threads": ["network"],
        "inverse_temperature": [],  # passed as placeholder
        "learning_rate": [],        # passed as placeholder
        "loss": ["network"],        # is used in gradients, model.loss, neuralnetwork.summary node wrong
        "max_steps": ["input"],
        "number_of_eigenvalues": [],# used only in python code
        "number_walkers": ["network"],
        "optimizer": ["network"],   # optimizer instance added to network
        "output_activation": ["network"],
        "output_dimension": ["input", "network"],
        "parse_parameters_file": ["network"],
        "parse_steps": ["network"],
        "prior_factor": ["network"],    # would need to reset "sampler.set_prior"
        "prior_lower_boundary": ["network"],
        "prior_power": ["network"],
        "prior_upper_boundary": ["network"],
        "progress": [],             # only used in python code
        "restore_model": ["network"],
        "run_file": ["network"],
        "sampler": ["network"],     # adds a specific sampler instance to network
        "save_model": ["network"],
        "seed": ["network"],
        "sigma": ["network"],       # served as values, might be replaced by placeholders
        "sigmaA": ["network"],      # served as values, might be replaced by placeholders
        "sql_db": [],               # affects only runtime, not used in simulation
        "step_width": [],           # fed via placeholder
        "summaries_path": ["network"],
        "trajectory_file": ["network"],
        "use_reweighting": [],      # only used in python code
        "verbose": [],              # only used in python code, log level changed on assignment by pythonoptions.set
    }

    def __init__(self, **kwargs):
        """Initializes the internal neural network and everything.

        Args:
          **kwargs: 

        Returns:

        """
        super(Simulation, self).__init__()
        self._nn = None

        # remove extra option dataset
        features,labels = None, None
        if "dataset" in kwargs.keys():
            features=kwargs["dataset"][0]
            labels=kwargs["dataset"][1]
            del kwargs["dataset"]

        # construct options object and initialize neuralnetwork
        self._options = PythonOptions(add_keys=True, value_dict=kwargs)
        logging.info(self._options)
        self._nn = Model(self._options)

        # if this assert triggers, then a new option has been added and its not
        # yet been stated what parts of the internal state of simulation
        # interface are affected, see `Simulation._affects_map`.
        # Solution to fix: Add an entry to this map with a list of what's affected
        assert( sorted(self._affects_map.keys()) == sorted(self._options._default_map.keys()) )

        if features is not None:
            self._nn.provide_data(features=features, labels=labels)
            self._lazy_nn_construction = False
        elif "batch_data_files" in kwargs.keys() \
                and (len(kwargs["batch_data_files"]) != 0):
            self._nn.init_input_pipeline()
            self._lazy_nn_construction = False
        else:
            self._lazy_nn_construction = True

        # we need to evaluate loss, gradients, hessian, accuracy, ... on the 
        # same batch. Hence, we cache the results here for one-time gets
        self._cache = EvaluationCache(self._nn)

        # construct nn if dataset has been provided
        self._construct_nn()

        self._parameters = Parameters(self._nn, ["weights", "biases"], self._cache)
        self._momenta = Parameters(self._nn, ["momenta_weights", "momenta_biases"])

        self.non_simplified_access = False

    @staticmethod
    def help(key=None):
        """Prints help for each option or all option names if key is None

        Args:
          key: name of option or None for list of options (Default value = None)

        Returns:

        """
        PythonOptions.help(key)

    def _construct_nn(self):
        """Constructs the neural network is dataset is present."""
        if not self._lazy_nn_construction:
            self._nn.init_network(None, setup="trainsample", add_vectorized_gradients=True)
            self._nn.reset_dataset()
            self._cache._init_node_keys()
            self._cache.reset()
            self._lazy_nn_construction = False

    def _check_nn(self):
        if self._nn is None:
            raise AttributeError("Neural network has not been constructed, dataset provided?")

    def set_options(self, **kwargs):
        """Resets some of the options to new values given by the keyword
        dictionary in `kwargs`.
        
        Warning:
            This may reset the dataset or even the network depending on what
            parameters are changed.

        Kwargs:
          any option listed in `Simulation.affects_map`

        Returns:
          None

        """
        # set the new option values
        self._options.set_options(**kwargs)

        # scan arguments and set flags for parts required to reset
        affected_parts = set()
        for key in kwargs.keys():
            for value in self._affects_map[key]:
                affected_parts.add(value)
        affected_parts = list(affected_parts)
        logging.info("Parts affected by change of options are "+str(affected_parts)+".")

        if "network" in affected_parts:
            raise ValueError("Changing the network is not yet supported.")

        if "network" in affected_parts and self._lazy_nn_construction:
            # Save network parameters
            values = self.parameters
            old_dimensions = [self._nn.FLAGS.input_dimension,
                              self._nn.FLAGS.hidden_dimension, self._nn.FLAGS.output_dimension]
            # reset the network and tensorflow's default graph
            self._nn = Model(self._options)
        if "input" in affected_parts:
            if self._options._option_map["in_memory_pipeline"]:
                features = self._nn.input_pipeline.features
                labels = self._nn.input_pipeline.labels
                self._nn.provide_data(features, labels)
            else:
                self._nn.init_input_pipeline()
        if "network" in affected_parts:
            # reconstruct the network
            was_lazy = self._lazy_nn_construction
            self._construct_nn()
            if not was_lazy:
                try:
                    self._reassign_parameters(values, old_dimensions)
                except ValueError:
                    logging.warning("New network has random starting parameters as it has less layers than the old one.")
        if "input" in affected_parts:
            self._nn.reset_dataset()

    def _reassign_parameters(self, values, dimensions):
        """Reassigns a parameter set of a possibly smaller or larger network

        Args:
          values: old weights and biases
          dimensions: dimensions of old network

        Returns:

        """
        new_values = NetworkParameterAdapter(values, dimensions,
                                [self.options.input_dimension]+ \
                                    self.options.hidden_dimensions+ \
                                [self.options.input_dimension])
        self._nn.assign_neural_network_parameters(new_values)

    def _evaluate_cache(self, key, walker_index=None):
        """Evaluates the neural network from the possibly cached values.

        Args:
          key: name of node
          walker_index: index of walker to evaluate or None for all (Default value = None)

        Returns:
          value for the given node and walker

        """
        if walker_index is None and self._nn.FLAGS.number_walkers == 1 \
            and not self.non_simplified_access:
            return self._cache.evaluate(key, walker_index)[0]
        else:
            return self._cache.evaluate(key, walker_index)

    def loss(self, walker_index=None):
        """Evalutes the current loss.

        Args:
          walker_index: index of walker to use or None for all (Default value = None)

        Returns:
          value of the loss function for walker `walker_index`

        """
        return self._evaluate_cache("loss", walker_index)

    def gradients(self, walker_index=None):
        """Evaluates the gradient of the loss with respect to the set
        of parameters at the current parameters.
        
        For sake of speed, the parameters have to be set beforehand.

        Args:
          walker_index: index of walker to use for fitting or None for all (Default value = None)

        Returns:
          gradients for walker `walker_index`

        """
        if not self._cache.hasNode("gradients"):
            raise AttributeError("Gradient nodes have not been added to the graph.")
        return self._evaluate_cache("gradients", walker_index)

    def hessians(self, walker_index=None):
        """Evaluates the hessian of the loss with respect to the
        set of parameters at the current parameters.
        
        For sake of speed, the parameters have to be set beforehand.

        Args:
          walker_index: index of walker to use for fitting or None for all (Default value = None)

        Returns:
          hessian for walker `walker_index`

        """
        if not self._cache.hasNode("hessians"):
            raise AttributeError("Hessian nodes have not been added to the graph." \
                                 +" You need to explicitly set 'do_hessians' to True in options.")
        return self._evaluate_cache("hessians", walker_index)

    def score(self, walker_index=None):
        """Evaluates the accuracy on the given dataset

        Args:
          walker_index: index of walker to use for fitting (Default value = None)

        Returns:
          accuracy for walker `walker_index`

        """
        return self._evaluate_cache("accuracy", walker_index)

    @property
    def parameters(self):
        """Returns the current set of parameters

        Returns:
            parameters

        """
        self._check_nn()
        if not self.non_simplified_access and len(self._parameters) == 1:
            return self._parameters[0]
        else:
            return self._parameters

    @parameters.setter
    def parameters(self, values):
        """Assigns the current parameters from `parameters`.
        
        The parameters are expected as a flat numpy array of the size
        of `simulation.num_parameters()`.

        Args:
          values: new parameters to set

        Returns:

        """
        self._check_nn()
        for i in range(len(self._parameters)):
            self._parameters[i] = values
            self._cache.invalidate_cache(i)

    @property
    def momenta(self):
        """Returns the current momentum to each parameter.

        Returns:
            momenta or None if sampler does not support momenta

        """
        self._check_nn()
        try:
            momenta = self._momenta
        except ValueError:
            logging.error("%s does not have momenta." % (self._options.sampler))
            return None
        if not self.non_simplified_access and len(momenta) == 1:
            return momenta[0]
        else:
            return momenta


    @momenta.setter
    def momenta(self, values):
        """Returns the current momentum to each parameter.

        Args:
          values: new momenta to set

        Returns:

        """
        self._check_nn()
        try:
            for i in range(len(self._momenta)):
                self._momenta[i] = values
        except ValueError:
            logging.error("%s does not have momenta." % (self._options.sampler))
            return None

    def init_momenta(self, inverse_temperature = None):
        """Reinitializes the network parameter's momenta from gaussian distribution
        with `inverse_temperature` as stddev and 0 mean.
        
        Note:
            This uses numpy's standard_normal to initialize.
            Set `numpy.random.seed()` to obtain reproducible runs.

        Args:
          inverse_temperature: inverse temperature for momenta scaling or None for default

        Returns:

        """
        if inverse_temperature is None:
            inverse_temperature = self._nn.FLAGS.inverse_temperature
        for i in range(len(self._parameters)):
            self.momenta = \
                np.random.standard_normal(size=(self.num_parameters()))*inverse_temperature

    def num_parameters(self):
        """Returns the number of parameters of the neural network.

        Returns:
            number of parameters/degrees of freedom of the network

        """
        self._check_nn()
        return self._nn.get_total_weight_dof() + self._nn.get_total_bias_dof()

    def num_walkers(self):
        """Returns the number of replicated copies of the neural network, i.e. walkers

        Returns:
            number of walkers/replicated copies of the network

        """
        self._check_nn()
        return self._nn.FLAGS.number_walkers

    def fit(self, walker_index=0):
        """Fits the parameters of the neural network to best match with the
        given dataset.
        
        Note that the parameters of the fit such as `optimizer`,
        `learning_rate` are all set in the `__init__()` options statement.

        Args:
          walker_index: index of walker to use for fitting (Default value = 0)

        Returns:
          TrajectoryData` containing run_info, trajectory, averages
          pandas dataframes

        """

        self._check_nn()
        self._nn.reset_dataset()
        run_info, trajectory, averages = \
            self._nn.train(return_run_info=True, \
                           return_trajectories=True,
                           return_averages=True)
        self._cache.reset()
        return TrajectoryData(run_info, trajectory, averages)

    def sample(self):
        """Performs sampling of the neural network's loss manifold for all walkers.
        
        Note:
            The parameters of the sampling such as `sampler`, `step_width`
            are all set in the `__init__()` options statement.
        
            At the moment, this function will perform sampling for all walkers
            at once.

        Returns:
            `TrajectoryData` containing run_info, trajectory, averages pandas
            dataframes

        """
        self._check_nn()
        self._nn.reset_dataset()
        run_info, trajectory, averages = \
            self._nn.sample(return_run_info=True, \
                            return_trajectories=True,
                            return_averages=True)
        self._cache.reset()
        return TrajectoryData(run_info, trajectory, averages)

    @property
    def dataset(self):
        """Getter for the dataset as a numpy array with respect to the
        currently chosen `batch_size`.

        Returns:
            array of features and labels, each a numpy array of `batch_size`

        """
        return self._cache.dataset

    @dataset.setter
    def dataset(self, value):
        """Evaluates accuracy on a new dataset `dataset`
        
        Note:
            This sets the `dataset` as the new dataset replacing the old one.

        Returns:
            accuracy for `dataset`

        """
        if isinstance(value, str) or \
                (isinstance(value, list) and isinstance(value[0], str)):
            # file name: parse
            if isinstance(value, str):
                self._nn.FLAGS.batch_data_files = [value]
            else:
                self._nn.FLAGS.batch_data_files = value
            self._nn.init_input_pipeline()
        elif isinstance(value, list):
            # is list of [features, labels]
            if len(value) != 2:
                raise TypeError("Dataset needs to contain both features and labels.")
            self._nn.provide_data(value[0], value[1])
        else:
            raise TypeError("Unknown dataset object which is neither file name nor list")
        if self._lazy_nn_construction:
            self._lazy_nn_construction = False
            self._construct_nn()
        self._nn.reset_dataset()
        self._cache.reset()

    def predict(self, features, walker_index=0):
        """Evaluates predictions (i.e. output of network) for the given features.

        Args:
          features: feature array to predict labels for
          walker_index: index of walker to use for prediction (Default value = 0)

        Returns:
          labels for `features` predicted by walker `walker_index`

        """
        self._check_nn()
        # set up feed_dict
        feed_dict = {self._nn.xinput: features}

        # evaluate the output "y" nodes
        y_node = self._nn.nn[walker_index].get_list_of_nodes(["y"])
        y_eval = self._nn.sess.run(y_node, feed_dict=feed_dict)
        return y_eval
