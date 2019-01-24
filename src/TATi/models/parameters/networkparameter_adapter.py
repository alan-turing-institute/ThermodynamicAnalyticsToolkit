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

from hmac import new

import numpy as np


class NetworkParameterAdapter(object):
    """Adapts parameters obtained from one network size to another.

    Args:
      When: one changes the network dimensions
      as: it internally needs to fulfill a certain structure
      This: class helps in adapting a parameter vector obtained from a network
      of: one size to the parameter vector of a network with a different size

    Returns:

    """

    perturbation_scale = 1e-2

    @staticmethod
    def convert(old_parameters, old_dimensions, new_dimensions):
        """Converts `old_parameters` conforming to the sizes defined in
        `old_dimensions` to `new_dimensions`.

        Args:
          old_parameters: numpy array of parameters
          old_dimensions: list of input, list of hidden, and output dimensions
          new_dimensions: list of input, list of hidden, and output dimensions

        Returns:
          new numpy array of parameters

        """
        print(old_parameters.shape)

        if len(old_dimensions) > len(new_dimensions):
            raise ValueError("We cannot remove layers from the network and maintain parameters.")

        new_params = []
        index_start = 0

        # convert each weight layer
        for i in range(1, len(old_dimensions)):
            index_end = index_start + old_dimensions[i-1]*old_dimensions[i]
            print("Resizing layer (%d, %d) to (%d, %d)" % (
                    old_dimensions[i - 1], old_dimensions[i],
                    new_dimensions[i - 1], new_dimensions[i]
            ))
            old_layer_weights = old_parameters[index_start:index_end]
            layer_weights = \
                NetworkParameterAdapter._convert_single_layer_weights(
                    old_layer_weights,
                    [old_dimensions[i-1], old_dimensions[i]],
                    [new_dimensions[i-1], new_dimensions[i]])
            new_params.append(layer_weights)
            index_start += index_end

        # add more weight layers if necessary
        for i in range(len(old_dimensions), len(new_dimensions)):
            layer_weights = NetworkParameterAdapter._add_diagonal_layer_weights(
                [new_dimensions[i - 1], new_dimensions[i]])
            new_params.append(layer_weights)

        # convert each bias layer
        for i in range(1, len(old_dimensions)):
            old_layer_biases = old_parameters[index_start:index_end]
            layer_biases = \
                NetworkParameterAdapter._convert_single_layer_biases(
                    old_layer_biases,
                    [new_dimensions[i-1], new_dimensions[i]])
            new_params.append(layer_biases)
            index_start += index_end

        # add more bias layers if necessary
        for i in range(len(old_dimensions), len(new_dimensions)):
            layer_biases = NetworkParameterAdapter._add_diagonal_layer_biases(
                [new_dimensions[i - 1], new_dimensions[i]])
            new_params.append(layer_biases)

        new_parameters = np.concatenate(new_params)
        print(new_parameters.shape)
        
        return new_parameters

    @staticmethod
    def _convert_single_layer_weights(layer_weights, old_dims, new_dims):
        """This converts a single set of weights from one layer having `old_dims`
        into `new_dims`

        Args:
          layer_weights: numpy array with old weights
          old_dims: list of input and output dimension of old network
          new_dims: list of input and output dimension of new network

        Returns:
          convert set of weights

        """
        # print(hidden_layer_weights)
        layer_weights.shape = (old_dims[0], old_dims[1])
        # print("Reshaped")
        # print(layer_weights)
        new_layer_weights = np.random.rand(new_dims[0], new_dims[1]) * NetworkParameterAdapter.perturbation_scale
        min_dims = [min(layer_weights.shape[i],new_layer_weights.shape[i]) for i in range(2)]
        new_layer_weights[:min_dims[0], :min_dims[1]] = layer_weights
        # print("Padded")
        # print(new_layer_weights)
        new_layer_weights.shape = (new_dims[0] * new_dims[1])
        # print("Flattened")
        # print(new_layer_weights)
        return new_layer_weights

    @staticmethod
    def _convert_single_layer_biases(layer_biases, new_dims):
        """This converts a single set of biases from one layer having `old_dims`
        into `new_dims`

        Args:
          layer_biases: numpy array with old biases
          new_dims: list of input and output dimension of new network

        Returns:
          convert set of biases

        """
        # print(layer_biases)
        ###layer_biases.shape = (old_dims[1])
        # print("Reshaped")
        # print(layer_biases)
        new_layer_biases = np.ones(new_dims[1]) * NetworkParameterAdapter.perturbation_scale
        min_dims = min(layer_biases.shape[0], layer_biases.shape[0])
        new_layer_biases[:min_dims] = layer_biases
        # print("Padded")
        # print(new_layer_biases)
        ###new_layer_biases.shape = (new_dims[1])
        # print("Flattened")
        # print(new_layer_biases)
        return new_layer_biases

    @staticmethod
    def _add_diagonal_layer_weights(dims):
        """This adds a new layer with ones on the diagonal and small values unequal to
        zero everywhere else, i.e. a pass-thru layer.

        Args:
          dims: list of input and output dimension of new network

        Returns:
          numpy array with new weight matrix

        """
        layer_weights = np.random.rand(dims[0], dims[1]) * NetworkParameterAdapter.perturbation_scale
        min_dim = min(dims[0], dims[1])
        # set ones on diagonal
        for i in range(min_dim):
            layer_weights[i][i] = 1.
        return layer_weights

    @staticmethod
    def _add_diagonal_layer_biases(dims):
        """This adds a new vector biases all small unequal to zero values

        Args:
          dims: list of input and output dimension of new network

        Returns:
          numpy array with new bias parameters

        """
        return np.ones(dims[1]) * NetworkParameterAdapter.perturbation_scale

