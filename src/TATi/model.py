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

import logging
from math import sqrt

import scipy.sparse as sps
from scipy.sparse import linalg

from TATi.models.input.inputpipelinefactory import InputPipelineFactory
from TATi.models.input.inmemorypipeline import InMemoryPipeline
from TATi.models.modelstate import ModelState
from TATi.options.pythonoptions import PythonOptions


class Model(object):
    """This class is the low-level interface to TATi, allowing access to
    the network, the input pipeline and so on.
    
    Its internal state is associated with `ModelState`. Here, we just
    define the API to interface the state.
    
    Note:
        Needs to be properly initialized after construction by calling
        `model.init_init_input_pipeline()` and subsequently(!)
        `model.init_network()`.
    
    Warning:
        It is important to always initialize the input pipeline *before*
        the network. TATi needs to know the feature vector dimension
        in order to correctly set up the input layer. See the Example
        below.
    
    Example:
        We briefly show the typical initialization and use of the class.
    
        >> from TATi.model import Model
        >> FLAGS = Model.setup_parameters()
        >> nn = Model(FLAGS)
        >> nn.init_input_pipeline()
        >> nn.init_network(setup="sample")
        >> nn.sample()

    Args:

    Returns:

    """
    def __init__(self, FLAGS):
        """Initialize the class with options in `FLAGS`.

        Args:
          FLAGS: options dict see class `Options`

        Returns:

        """
        self.state = ModelState(FLAGS)

    def reset(self):
        """Reset the internal state of the model.
        
        WARNING:
            This destroys the whole computational graph and
            resets _all_ of the state's member variables.

        Args:

        Returns:

        """
        self.state.reset()

    @property
    def FLAGS(self):
        """Getter for the options structure that specifies the network,
        how to sample/train, and everything.

        Args:

        Returns:
          FLAGS, see `Options`

        """
        return self.state.FLAGS

    @property
    def nn(self):
        """Getter for the `NeuralNetwork` object containing weight
        and bias nodes and everything network related.

        Args:

        Returns:
          `NeuralNetwork` object

        """
        return self.state.nn

    @property
    def sess(self):
        """Getter for the tensorflow *Session* object that is required to
        evaluate nodes in the computational graph.

        Args:

        Returns:
          `tf.Session` object

        """
        return self.state.sess

    @property
    def input_pipeline(self):
        """Getter for the `InputPipeline` object.
        
        Pipeline may be specialized depending on `model.FLAGS()`.

        Args:

        Returns:
          `InputPipeline` object

        """
        return self.state.input_pipeline

    @property
    def xinput(self):
        """Getter for the input layer `tf.placeholder` object.
        
        These are needed for the *feed_dict* mechanism to feed the features
        as input into the neural network.

        Args:

        Returns:
          input layer placeholders

        """
        return self.state.xinput

    @property
    def true_labels(self):
        """Getter for the output layer `tf.placeholder` object.
        
        These are needed for the *feed_dict* mechanism to feed the labels
        as output into the neural network.

        Args:

        Returns:
          output layer placeholders

        """
        return self.state.true_labels

    @property
    def weights(self):
        """Getter for the linearized weight parameters of the neural network.

        Args:

        Returns:
          weight `neuralnet_parameters` object

        """
        return self.state.weights

    @property
    def biases(self):
        """Getter for the linearized bias parameters of the neural network.
        
        Args:

        Returns:
          bias `neuralnet_parameters` object

        """
        return self.state.biases

    @property
    def momenta_weights(self):
        """Getter for the linearized weight momenta of the neural network.
        
        WARNING:
            These are only available during sampling and only when a *second
            order* sampler is employed (i.e. not SGLD).
        
        Args:

        Returns:
          weight momenta `neuralnet_parameters` object

        """
        return self.state.momenta_weights

    @property
    def momenta_biases(self):
        """Getter for the linearized bias momenta of the neural network.
        
        WARNING:
            These are only available during sampling and only when a *second
            order* sampler is employed (i.e. not SGLD).

        Args:

        Returns:
          bias momenta `neuralnet_parameters` object

        """
        return self.state.momenta_biases

    @property
    def loss(self):
        """Getter for the loss node of the neural network.
        
        This node can be used to request the current loss from the
        computational graph.

        Args:

        Returns:
          loss node, see `tf.losses`

        """
        return self.state.loss

    @property
    def gradients(self):
        """Getter for the linearized gradients node of the neural network,
        i.e. all linearized weights and all linearized differentiated with
        respect to the loss, see `model.loss()`.

        Args:

        Returns:
          gradient node

        """
        return self.state.gradients

    @property
    def hessians(self):
        """Getter for the hessian matrix node of the neural network,
        i.e. all linearized weights and all linearized twice differentiated
        with respect to the loss, see `model.loss()`.

        Args:

        Returns:
          hessian node

        """
        return self.state.hessians

    def init_input_pipeline(self):
        """Initializes the input pipeline."""
        self.state.init_input_pipeline()

    def init_network(self, filename=None, setup=None, add_vectorized_gradients=False):
        """Initializes the graph, from a stored model if filename is not None.

        Args:
          filename: name of file containing stored model (Default value = None)
          setup: sample", "train" or else to add nodes that trigger a
        single sampling or training step. Otherwise they are not added.
        init_network() can be called consecutively with both variants
        to add either type of node. (Default value = None)
          add_vectorized_gradients: add nodes to return gradients in fully
        vectorized form, i.e. in the same sequence as nn_weights and
        nn_biases parameters combined, see self.gradients. (Default value = False)

        Returns:

        """
        self.state.init_network(filename=filename, setup=setup,
                                add_vectorized_gradients=add_vectorized_gradients)

    def get_averages_header(self, setup):
        """Return header for the averages CSV file.

        Args:
          setup: either "sample" or "train"

        Returns:
          averages header string

        """
        if setup == "sample":
            return self.state.trajectory_sample.get_averages_header()
        elif setup == "train":
            return self.state.trajectory_train.get_averages_header()

    def get_run_header(self, setup):
        """Return header for the run info CSV file.

        Args:
          setup: either "sample" or "train"

        Returns:
          run info header string

        """
        if setup == "sample":
            return self.state.trajectory_sample.get_run_header()
        elif setup == "train":
            return self.state.trajectory_train.get_run_header()

    def get_parameters(self):
        """Getter for the internal set oF FLAGS controlling training and sampling.

        Args:

        Returns:
          FLAGS parameter set

        """
        return self.state.FLAGS

    def reset_parameters(self, FLAGS):
        """Use to pass a different set of FLAGS controlling training or sampling.

        Args:
          FLAGS: FLAGS dict, see `Options`

        Returns:
          new set of parameters

        """
        self.state.reset_parameters(FLAGS)

    @staticmethod
    def setup_parameters(**kwargs):
        """Creates a default *FLAGS* structure containing all options that
        control the network, sampling/training and so on.

        Args:
          kwargs: dict to override default values by keyword, value pairs, see `PythonOptions._description_map`

        Returns:
          created *FLAGS* dict

        """
        return PythonOptions(add_keys=True, value_dict=kwargs)

    def reset_dataset(self):
        """Re-initializes the dataset for a new run.
        """
        self.state.reset_dataset()

    def get_total_weight_dof(self):
        """Returns the total number of weight parameters or weight degrees of
        freedom of the network.

        Args:

        Returns:
          number of weight parameters

        """
        return self.state.weights[0].get_total_dof()

    def get_total_bias_dof(self):
        """Returns total number of bias parameters or bias degrees of
        freedom of the network.

        Args:

        Returns:
          number of bias parameters

        """
        return self.state.biases[0].get_total_dof()

    def compute_optimal_stepwidth(self, walker_index=0):
        """Computes the optimal stepwidth at the current loss manifold point
        by inspecting the hessian matrix.
        
        Warning:
            This is potentially very costly as the dominant eigenvalues of the
            hessian need to be calculated.
        
            Moreover, setting up the hessian is \f$O(N^2)\f$ in the number of
            parameters N, too. Therefore, construction of the hessian node
            needs to explicitly enabled in the *FLAGS* by setting `do_hessians`
            to True.

        Args:
          walker_index: index of the walker for whose position to calculate (Default value = 0)

        Returns:
          optimal step width

        """
        assert(walker_index < self.state.FLAGS.number_walkers)
        placeholder_nodes = self.state.nn[walker_index].get_dict_of_nodes(["learning_rate", "y_"])

        # get first batch of data
        self.reset_dataset()
        features, labels = self.state.input_pipeline.next_batch(self.state.sess)

        # place in feed dict
        feed_dict = {
            self.state.xinput: features,
            placeholder_nodes["y_"]: labels,
            placeholder_nodes["learning_rate"]: self.state.FLAGS.learning_rate
        }
        if self.state.FLAGS.dropout is not None:
            feed_dict.update({placeholder_nodes["keep_prob"]: self.state.FLAGS.dropout})

        hessian_eval = self.state.sess.run(self.state.hessians[walker_index], feed_dict=feed_dict)
        lambdas, _ = sps.linalg.eigs(hessian_eval, k=1)
        optimal_step_width = 2/sqrt(lambdas[0])
        logging.info("Optimal step width would be "+str(optimal_step_width))

    def save_model(self, filename):
        """Saves the current neural network model to a set of files,
        whose prefix is given by filename.
        
        See also `model.restore_model()`.

        Args:
          filename: prefix of set of model files

        Returns:
          path where model was saved

        """
        return self.state.save_model(filename)

    def restore_model(self, filename):
        """Restores the model from a tensorflow checkpoint file.
        
        Compare to `model.save_model()`.

        Args:
          filename: prefix of set of model files

        Returns:

        """
        self.state.restore_model(filename)

    def provide_data(self, features, labels, shuffle=False):
        """Use to provide an in-memory dataset, i.e., numpy arrays with
        `features` and `labels`.

        Args:
          features: feature part of dataset
          labels: label part of dataset
          shuffle: whether to shuffle the dataset initially or not (Default value = False)

        Returns:

        """
        self.state.provide_data(features=features, labels=labels, shuffle=shuffle)

    def create_model_file(self, initial_step, parameters, model_filename):
        """Create a tensorflow checkpoint model file from a given step
        number in `initial_step` and parameters in `parameters`.

        Args:
          initial_step: step number to place in model
          parameters: parameters to place in model
          model_filename: filename prefix of model

        Returns:

        """
        self.assign_current_step(initial_step)
        self.assign_neural_network_parameters(parameters)
        self.save_model(model_filename)

    def assign_current_step(self, step, walker_index=0):
        """Allows to set the current step number of the iteration.

        Args:
          step: step number to set
          walker_index: walker for which to set step (Default value = 0)

        Returns:

        """
        self.state.assign_current_step(step, walker_index=walker_index)

    def assign_neural_network_parameters(self, parameters, walker_index=0, do_check=False):
        """Assigns the parameters of the neural network from
        the given array.

        Args:
          parameters: list of values, one for each weight and bias
          walker_index: index of the replicated network (in the graph) (Default value = 0)
          do_check: whether to check set values (and print) or not

        Returns:
          evaluated weights and bias on do_check or None otherwise (Default value = False)

        """
        weights_dof = self.state.weights[walker_index].get_total_dof()
        return self.state.assign_weights_and_biases(weights_vals=parameters[0:weights_dof],
                                                    biases_vals=parameters[weights_dof:],
                                                    walker_index=walker_index,
                                                    do_check=do_check)

    def assign_weights_and_biases(self, weights_vals, biases_vals, walker_index=0, do_check=False):
        """Assigns weights and biases of a neural network.

        Args:
          weights_vals: flat weights parameters
          biases_vals: flat bias parameters
          walker_index: index of the replicated network (in the graph) (Default value = 0)
          do_check: whether to check set values (and print) or not

        Returns:
          evaluated weights and bias on do_check or None otherwise (Default value = False)

        """
        return self.state.assign_weights_and_biases(weights_vals=weights_vals,
                                                    biases_vals=biases_vals,
                                                    walker_index=walker_index,
                                                    do_check=do_check)

    def assign_weights_and_biases_from_dataframe(self, df_parameters, rownr, walker_index=0, do_check=False):
        """Parse weight and bias values from a dataframe given a specific step
        to set the neural network's parameters.

        Args:
          df_parameters: pandas dataframe
          rownr: rownr to set
          walker_index: index of the replicated network (in the graph) (Default value = 0)
          do_check: whether to evaluate (and print) set parameters

        Returns:
          evaluated weights and bias on do_check or None otherwise (Default value = False)
        """
        # check that column names are in order
        return self.state.assign_weights_and_biases_from_dataframe(
            df_parameters=df_parameters, rownr=rownr, walker_index=walker_index, do_check=do_check)

    def assign_weights_and_biases_from_file(self, filename, step, walker_index=0, do_check=False):
        """Parse weight and bias values from a CSV file given a specific step
        to set the neural network's parameters.

        Args:
          filename: filename to parse
          step: step to set (i.e. value in "step" column designates row)
          walker_index: index of the replicated network (in the graph) (Default value = 0)
          do_check: whether to evaluate (and print) set parameters

        Returns:
          evaluated weights and bias on do_check or None otherwise (Default value = False)

        """
        return self.state.assign_weights_and_biases_from_file(
            filename=filename, step=step, walker_index=walker_index, do_check=do_check)

    def sample(self, return_run_info=False, return_trajectories=False, return_averages=False):
        """Performs the actual sampling of the neural network `nn` given a dataset `ds` and a
        Session `session`.

        Args:
          return_run_info: if set to true, run information will be accumulated
        inside a numpy array and returned (Default value = False)
          return_trajectories: if set to true, trajectories will be accumulated
        inside a numpy array and returned (adhering to FLAGS.every_nth) (Default value = False)
          return_averages: if set to true, accumulated average values will be
        returned as numpy array (Default value = False)

        Returns:
          either thrice None or lists (per walker) of pandas dataframes
          depending on whether either parameter has evaluated to True

        """
        return self.state.execute_trajectory_run(self.state.trajectory_sample,
                                                 return_run_info, return_trajectories, return_averages)

    def train(self, return_run_info=False, return_trajectories=False, return_averages=False):
        """Performs the actual training of the neural network `nn` given a dataset `ds` and a
        Session `session`.

        Args:
          return_run_info: if set to true, run information will be accumulated
        inside a numpy array and returned (Default value = False)
          return_trajectories: if set to true, trajectories will be accumulated
        inside a numpy array and returned (adhering to FLAGS.every_nth) (Default value = False)
          return_averages: if set to true, accumulated average values will be
        returned as numpy array (Default value = False)

        Returns:
          either twice None or a pandas dataframe depending on whether either
          parameter has evaluated to True

        """
        return self.state.execute_trajectory_run(self.state.trajectory_train,
                                                 return_run_info, return_trajectories, return_averages)

    def deactivate_file_writing(self):
        """Deactivates writing of average, run info and trajectory CSV file
        during `model.sample()` and `model.train()`.

        Args:

        Returns:

        """
        for entity in [self.state.trajectory_train, self.state.trajectory_sample]:
            if entity is not None:
                entity.set_config_map("do_write_averages_file", False)
                entity.set_config_map("do_write_run_file", False)
                entity.set_config_map("do_write_trajectory_file", False)
