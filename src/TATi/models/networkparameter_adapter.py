from hmac import new

import numpy as np


class NetworkParameterAdapter(object):
    """ Adapts parameters obtained from one network size to another.

    Parameters of a neural network can be obtained as a single numpy vector.
    When one changes the network dimensions, this vector is no longer valid
    as it internally needs to fulfill a certain structure.

    This class helps in adapting a parameter vector obtained from a network
    of one size to the parameter vector of a network with a different size.
    """

    perturbation_scale = 1e-2

    @staticmethod
    def convert(old_parameters, old_dimensions, new_dimensions):
        """ Converts `old_parameters` conforming to the sizes defined in
        `old_dimensions` to `new_dimensions`.

        :param old_parameters: numpy array of parameters
        :param old_dimensions: list of input, list of hidden, and output dimensions
        :param new_dimensions: list of input, list of hidden, and output dimensions
        :return: new numpy array of parameters
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
        """ This converts a single set of weights from one layer having `old_dims`
        into `new_dims`

        :param layer_weights: numpy array with old weights
        :param old_dims: list of input and output dimension of old network
        :param new_dims: list of input and output dimension of new network
        :return: convert set of weights
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
        """ This converts a single set of biases from one layer having `old_dims`
        into `new_dims`

        :param layer_biases: numpy array with old biases
        :param new_dims: list of input and output dimension of new network
        :return: convert set of biases
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
        """ This adds a new layer with ones on the diagonal and small values unequal to
        zero everywhere else, i.e. a pass-thru layer.

        :param dims: list of input and output dimension of new network
        :return: numpy array with new weight matrix
        """
        layer_weights = np.random.rand(dims[0], dims[1]) * NetworkParameterAdapter.perturbation_scale
        min_dim = min(dims[0], dims[1])
        # set ones on diagonal
        for i in range(min_dim):
            layer_weights[i][i] = 1.
        return layer_weights

    @staticmethod
    def _add_diagonal_layer_biases(dims):
        """ This adds a new vector biases all small unequal to zero values

        :param dims: list of input and output dimension of new network
        :return: numpy array with new bias parameters
        """
        return np.ones(dims[1]) * NetworkParameterAdapter.perturbation_scale

