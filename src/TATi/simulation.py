"""@package docstring
The simulation module contains the interface to generically access neural networks.

"""

import logging

from TATi.models.model import model
from TATi.options.pythonoptions import PythonOptions
from TATi.parameters.parameters import Parameters


class Simulation(object):
    """ This class represents the Python interface to TATi that allows to
    access the neural network (including its loss function, parameters,
    ...) as a black-box function.

    The idea is that there is no need to worry about any of the neural
    network internals. An initial parameter structure is all that is
    needed and afterwards one may treat the whole thing as two (coupled)
    functions, namely the loss and the predictor, where the loss depends
    implicitly on the dataset and both use the set of parameters of the
    neural network.
    """

    ### gives which parts of the interal state are affected by each option
    _affects_map = {
        "averages_file": ["network"],
        "batch_data_files": ["input", "network"],
        "batch_data_file_type": ["input", "network"],
        "batch_size": ["input"],
        "burn_in_steps": [],
        "collapse_after_steps": [],
        "covariance_blending": [],
        "diffusion_map_method": [],
        "do_hessians": ["network"],
        "dropout": ["network"],
        "every_nth": [],
        "fix_parameters": ["network"],
        "friction_constant": [],
        "hamiltonian_dynamics_time": [],
        "hidden_activation": ["network"],
        "hidden_dimension": ["network"],
        "in_memory_pipeline": ["input"],
        "input_columns": ["input"],
        "input_dimension": ["input", "network"],
        "inter_ops_threads": [],
        "intra_ops_threads": [],
        "inverse_temperature": [],
        "learning_rate": [],
        "loss": ["loss"],
        "max_steps": ["input"],
        "number_of_eigenvalues": [],
        "number_walkers": ["network"],
        "optimizer": [],
        "output_activation": ["network"],
        "output_dimension": ["input", "network"],
        "parse_parameters_file": ["network"],
        "parse_steps": ["network"],
        "prior_factor": [],
        "prior_lower_boundary": [],
        "prior_power": [],
        "prior_upper_boundary": [],
        "progress": [],
        "restore_model": ["network"],
        "run_file": ["network"],
        "sampler": [],
        "save_model": ["network"],
        "seed": ["network"],
        "sigma": [],
        "sigmaA": [],
        "sql_db": [],
        "step_width": [],
        "summaries_path": ["network"],
        "trajectory_file": ["network"],
        "use_reweighting": [],
        "verbose": [],
    }

    def __init__(self, **kwargs):
        """ Initializes the internal neural network and everything.

        :param **kwargs: keyword argument that set non-default values
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
        self._nn = model(self._options)

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
        self._cache = {}
        self._node_keys = {}

        # construct nn if dataset has been provided
        self._construct_nn()

        self._parameters = Parameters(self._nn)

    def _init_node_keys(self):
        """ Initializes the set of cached variables with nodes from the tensorflow's
        graph.

        Note:
            As we may initialize first after having set up the network, this
            needs to go into an extra function.
        """
        # TODO: This still needs to be adapted for multiple walkers, i.e. made into lists
        self._node_keys = {
                "loss": self._nn.loss,
                "accuracy": self._nn.nn[0].get_list_of_nodes(["accuracy"])
            }
        # add hessians and gradients only when nodes are present
        if self._nn.gradients is not None:
            self._node_keys["gradients"] = self._nn.gradients
        elif "gradients" in self._node_keys.keys():
            del self._node_keys["gradients"]
        if self._nn.hessians is not None:
            self._node_keys["hessians"] = self._nn.hessians
        elif "hessians" in self._node_keys.keys():
            del self._node_keys["hessians"]

    def _reset_cache(self):
        """ This resets the cache to None for all cached variables.
        """
        for key in self._node_keys.keys():
            self._cache[key] = None

    def _construct_nn(self):
        """ Constructs the neural network is dataset is present.
        """
        if not self._lazy_nn_construction:
            self._nn.init_network(None, setup="trainsample", add_vectorized_gradients=True)
            self._nn.reset_dataset()
            self._init_node_keys()
            self._reset_cache()

    def _check_nn(self):
        if self._nn is None:
            raise AttributeError("Neural network has not been constructed, dataset provided?")

    def set_options(self, **kwargs):
        """ Resets some of the options to new values given by the keyword
        dictionary in `kwargs`.

        Note:
            This may reset the dataset or even the network depending on what 
            parameters are changed.

        :param args: positional arguments
        :param **kwargs: keyword arguments
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
        if "input" in affected_parts:
            if self._options._option_map["in_memory_pipeline"]:
                features = self._nn.input_pipeline.features
                labels = self._nn.input_pipeline.labels
                self._nn.provide_data(features, labels)
            else:
                self._nn.init_input_pipeline()
            self._nn.reset_dataset()


    def _return_cache(self, key):
        """ This returns the cached element named `key` and resets the entry.

        :param key: key of cached variable
        :return: values stored in cache for `key`
        """
        assert( key in self._cache.keys() )
        print(self._cache)
        value = self._cache[key]
        self._cache[key] = None
        return value

    def _check_cache(self, key):
        """ Checks whether a value to `key` is in cache at the moment

        :param key: name to cached variable
        :return: True - value is cached, False - None is stored in cache
        """
        return self._cache[key] is not None

    def _update_cache(self):
        """ Updates the contents of the cache from the network's values.
        """
        nodes = list(self._node_keys.values())
        print(nodes)
        values = self._evaluate(nodes)
        print(values)
        for key, value in zip(self._node_keys.keys(), values):
            print("Setting "+key+" to "+str(value))
            self._cache[key] = value

    def _evaluate(self, nodes):
        """ Helper function to evaluate an arbitrary node of tensorflow's
        internal graph.

        :param nodes: node to evaluate in `session.run()`
        :return: result of node evaluation
        """
        self._check_nn()
        features, labels = self._nn.input_pipeline.next_batch(
            self._nn.sess, auto_reset=True)
        feed_dict = {
            self._nn.xinput: features,
            self._nn.nn[0].placeholder_nodes["y_"]: labels}
        eval_nodes = self._nn.sess.run(nodes, feed_dict=feed_dict)
        return eval_nodes

    def _evaluate_cache(self, key):
        """ Evaluates the variable `key` either from cache or updating if not
        present there.

        :param key: key of variable
        :return: valueto current batch of `key`
        """
        self._check_nn()
        if not self._check_cache(key):
            self._update_cache()
        return self._return_cache(key)

    def loss(self, walker_index=0):
        """ Evalutes the current loss.

        :param walker_index: index of walker to use for fitting
        :return: value of the loss function for walker `walker_index`
        """
        return self._evaluate_cache("loss")[walker_index]

    def gradients(self, walker_index=0):
        """ Evaluates the gradient of the loss with respect to the set
        of parameters at the current parameters.

        For sake of speed, the parameters have to be set beforehand.

        :param walker_index: index of walker to use for fitting
        :return: gradients for walker `walker_index`
        """
        if "gradients" not in self._node_keys.keys():
            raise AttributeError("Gradient nodes have not been added to the graph.")
        return self._evaluate_cache("gradients")[walker_index]

    def hessians(self, walker_index=0):
        """ Evaluates the hessian of the loss with respect to the
        set of parameters at the current parameters.

        For sake of speed, the parameters have to be set beforehand.

        :param walker_index: index of walker to use for fitting
        :return: hessian for walker `walker_index`
        """
        if "hessians" not in self._node_keys.keys():
            raise AttributeError("Hessian nodes have not been added to the graph." \
                                 +" You need to explicitly set 'do_hessians' to True in options.")
        return self._evaluate_cache("hessians")[walker_index]

    def score(self, walker_index=0):
        """ Evaluates the accuracy on the given dataset

        :param walker_index: index of walker to use for fitting
        :return: accuracy for walker `walker_index`
        """
        return self._evaluate_cache("accuracy")[walker_index]

    @property
    def parameters(self):
        """ Returns the current set of parameters

        :return: parameters
        """
        self._check_nn()
        return self._parameters

    @parameters.setter
    def parameters(self, values):
        """ Assigns the current parameters from `parameters`.

        The parameters are expected as a flat numpy array of the size
        of `simulation.num_parameters()`.

        :param values: new parameters to set
        """
        self._check_nn()
        for i in range(self._parameters.num_walkers()):
            self._parameters[i] = values

    @property
    def momenta(self):
        """ Returns the current momentum to each parameter.

        :return: momenta
        """
        raise NotImplementedError("Momenta are not yet implemented.")

    @momenta.setter
    def momenta(self, values):
        """ Returns the current momentum to each parameter.

        :param values: new momenta to set
        """
        raise NotImplementedError("Momenta are not yet implemented.")

    def num_parameters(self):
        """ Returns the number of parameters of the neural network.

        :return: number of parameters/degrees of freedom of the network
        """
        self._check_nn()
        return self._nn.get_total_weight_dof() + self._nn.get_total_bias_dof()

    def fit(self, walker_index=0):
        """ Fits the parameters of the neural network to best match with the
        given dataset.

        Note that the parameters of the fit such as `optimizer`,
        `learning_rate` are all set in the `__init__()` options statement.

        :param walker_index: index of walker to use for fitting
        :return: run_info, trajectory, averages
        """

        self._check_nn()
        self._nn.reset_dataset()
        run_info, trajectory, averages = \
            self._nn.train(walker_index,
                           return_run_info=True, \
                           return_trajectories=True,
                           return_averages=True)
        return run_info, trajectory, averages

    def sample(self):
        """ Performs sampling of the neural network's loss manifold.

        NOTE:
            The parameters of the sampling such as `sampler`, `step_width`
            are all set in the `__init__()` options statement.

        NOTE:
            At the moment, this function will perform sampling for all walkers
            at once.

        :return: run_info, trajectory, averages
        """
        self._check_nn()
        self._nn.reset_dataset()
        run_info, trajectory, averages = \
            self._nn.sample(return_run_info=True, \
                            return_trajectories=True,
                            return_averages=True)
        return run_info, trajectory, averages

    @property
    def dataset(self):
        """ Getter for the dataset as a numpy array with respect to the
        currently chosen `batch_size`.

        :return: array of features and labels, each a numpy array of `batch_size`
        """
        #self._check_nn()
        if self._nn is not None:
            return self._nn.input_pipeline.next_batch(
                self._nn.sess, auto_reset=True)
        else:
            return None

    @dataset.setter
    def dataset(self, value):
        """ Evaluates accuracy on a new dataset `dataset`

        NOTE: This sets the `dataset` as the new dataset replacing the old
        one.

        :return: accuracy for `dataset`
        """
        print("set")
        if isinstance(value, str) or \
                (isinstance(value, list) and isinstance(value[0], str)):
            # file name: parse
            if isinstance(value, str):
                self._nn.FLAGS.batch_data_files = [value]
            else:
                self._nn.FLAGS.batch_data_files = value
            self._nn.scan_dataset_dimension()
            self._nn.create_input_pipeline(self._nn.FLAGS)
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

    def predict(self, features, walker_index=0):
        """ Evaluates predictions (i.e. output of network) for the given features.

        :param: features - features to evaluate for network of walker `walker_index`
        :return: labels for `features` predicted by walker `walker_index`
        """
        self._check_nn()
        # set up feed_dict
        feed_dict = {self._nn.xinput: features}

        # evaluate the output "y" nodes
        y_node = self._nn.nn[walker_index].get_list_of_nodes(["y"])
        y_eval = self._nn.sess.run(y_node, feed_dict=feed_dict)
        return y_eval
