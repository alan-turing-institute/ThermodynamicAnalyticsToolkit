import logging
import numpy as np


class NetworkParameterAdapter(object):
    """ Functor that adapts parameters obtained from one network size to another.

    Parameters of a neural network can be obtained as a single numpy vector.
    When one changes the network dimensions, this vector is no longer valid
    as it internally needs to fulfill a certain structure.

    This class helps in adapting a parameter vector obtained from a network
    of one size to the parameter vector of a network with a different size.

    Note that this is only useful when either enlarging a hidden layer or when
    adding more hidden layers with linear activation functions.
    """

    def __init__(self, perturbation_scale = 1e-2):
        """ Cstor of the class that sets the perturbation scale

        Note:
            Weights are initialized randomly drawn from [0,1] times the scale,
            while biases are set to 0.1 times the scale.

        :param perturbation_scale: scale of the newly added components
        """
        super(NetworkParameterAdapter, self).__init__()
        self.perturbation_scale = perturbation_scale

    def __call__(self, old_parameters, old_dimensions, new_dimensions):
        """ Converts `old_parameters` conforming to the sizes defined in
        `old_dimensions` to `new_dimensions`.

        :param old_parameters: numpy array of parameters
        :param old_dimensions: list of input, hidden, and output dimensions
        :param new_dimensions: list of input, hidden, and output dimensions
        :return: new numpy array of parameters
        """
        #print(len(old_parameters))

        if len(old_dimensions) > len(new_dimensions):
            raise ValueError("We cannot remove layers from the network and maintain parameters.")

        new_params = []

        def resize_weight_matrix(index_begin, old_dim_index, new_dim_index):
            index_end = index_begin + old_dimensions[old_dim_index - 1] * old_dimensions[old_dim_index]
            logging.info("Resizing weight matrix (%d, %d) to (%d, %d)" % (
                old_dimensions[old_dim_index - 1], old_dimensions[old_dim_index],
                new_dimensions[new_dim_index - 1], new_dimensions[new_dim_index]
            ))
            old_layer_weights = np.array(old_parameters[index_begin:index_end])
            layer_weights = \
                self._convert_single_layer_weights(
                    old_layer_weights,
                    [old_dimensions[old_dim_index - 1], old_dimensions[old_dim_index]],
                    [new_dimensions[new_dim_index - 1], new_dimensions[new_dim_index]])
            new_params.append(layer_weights)
            return index_end

        def resize_bias_vector(index_begin, old_dim_index, new_dim_index):
            index_end = index_begin + old_dimensions[old_dim_index]
            logging.info("Resizing bias vector (%d) to (%d)" % (
                    old_dimensions[old_dim_index],
                    new_dimensions[new_dim_index]
            ))
            old_layer_biases = np.array(old_parameters[index_begin:index_end])
            layer_biases = \
                self._convert_single_layer_biases(
                    old_layer_biases,
                    [new_dimensions[new_dim_index - 1], new_dimensions[new_dim_index]])
            new_params.append(layer_biases)
            return index_end

        index_start = 0

        # convert each hidden weight layer
        for i in range(1, len(old_dimensions)-1):
            index_start = resize_weight_matrix(index_start, i, i)

        # add more weight layers if necessary
        for i in range(len(old_dimensions)-1, len(new_dimensions)-1):
            logging.info("Adding new weight matrix (%d, %d)" %(
                new_dimensions[i - 1], new_dimensions[i]
            ))
            layer_weights = self._add_diagonal_layer_weights(
                [new_dimensions[i - 1], new_dimensions[i]])
            new_params.append(layer_weights)

        # convert output weight layer
        index_start = resize_weight_matrix(index_start,
            len(old_dimensions)-1, len(new_dimensions)-1)

        # convert each hidden bias vector
        for i in range(1, len(old_dimensions)-1):
            index_start = resize_bias_vector(index_start, i, i)

        # add more bias vector if necessary
        for i in range(len(old_dimensions)-1, len(new_dimensions)-1):
            logging.info("Adding new bias vector (%d)" %(
                new_dimensions[i]
            ))
            layer_biases = self._add_diagonal_layer_biases(
                [new_dimensions[i - 1], new_dimensions[i]])
            new_params.append(layer_biases)

        # convert output bias vector
        index_start = resize_bias_vector(index_start,
                                         len(old_dimensions)-1, len(new_dimensions)-1)

        new_parameters = np.concatenate(new_params)

        return new_parameters

    def _convert_single_layer_weights(self, layer_weights, old_dims, new_dims):
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
        new_layer_weights = np.random.rand(new_dims[0], new_dims[1]) * self.perturbation_scale
        min_dims = [min(layer_weights.shape[i],new_layer_weights.shape[i]) for i in range(2)]
        new_layer_weights[:min_dims[0], :min_dims[1]] = layer_weights
        # print("Padded")
        # print(new_layer_weights)
        new_layer_weights.shape = (new_dims[0] * new_dims[1])
        # print("Flattened")
        # print(new_layer_weights)
        return new_layer_weights

    def _convert_single_layer_biases(self, layer_biases, new_dims):
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
        new_layer_biases = np.ones(new_dims[1]) * self.perturbation_scale
        new_layer_biases[:layer_biases.shape[0]] = layer_biases
        # print("Padded")
        # print(new_layer_biases)
        ###new_layer_biases.shape = (new_dims[1])
        # print("Flattened")
        # print(new_layer_biases)
        return new_layer_biases

    def _add_diagonal_layer_weights(self, dims):
        """ This adds a new layer with ones on the diagonal and small values unequal to
        zero everywhere else, i.e. a pass-thru layer.

        :param dims: list of input and output dimension of new network
        :return: numpy array with new weight matrix
        """
        layer_weights = np.random.rand(dims[0], dims[1]) * self.perturbation_scale
        min_dim = min(dims[0], dims[1])
        # set ones on diagonal
        for i in range(min_dim):
            layer_weights[i][i] = 1.
        layer_weights.shape = (dims[0] * dims[1])
        return layer_weights

    def _add_diagonal_layer_biases(self, dims):
        """ This adds a new vector biases all small unequal to zero values

        :param dims: list of input and output dimension of new network
        :return: numpy array with new bias parameters
        """
        return np.ones(dims[1]) * self.perturbation_scale

