import itertools

class Parameters(object):
    """ This class overrides `__getitem__` and `__setitem__` to allow simple
    access to the parameters of each walker.

    """

    def __init__(self, nn):
        self._nn = nn

    def __len__(self):
        """ Returns the number of parameters.

        :return: number of parameters
        """
        return self._nn.get_total_weight_dof() + self._nn.get_total_bias_dof()

    def num_walkers(self):
        """ Returns the number of walkers, i.e. the number of parameter sets
        of same length.

        :return: number of walkers/parameter sets
        """
        return self._nn.FLAGS.number_walkers

    def _valid_walker_index(self, walker_index):
        """ Ascertains that `walker_index` is in valid range.

        :param walker_index: index of walker
        :return: True - walker index is valid, False - not
        """
        assert( 0 <= walker_index < self.num_walkers() )

    def __getitem__(self, walker_index):
        """ Returns the parameters of a specific walker `walker_index`.

        :param walker_index: Index of the walker
        :return: parameter set of the respective walker
        """
        self._valid_walker_index(walker_index)
        weights_eval = self._nn.weights[walker_index].evaluate(self._nn.sess)
        biases_eval = self._nn.biases[walker_index].evaluate(self._nn.sess)
        return list(itertools.chain(weights_eval, biases_eval))


    def __setitem__(self, walker_index, parameters):
        """ Sets the parameters for a specific walker

        :param walker_index: Index of the walker
        :param parameters: set of new parameters
        """
        self._valid_walker_index(walker_index)
        assert(len(parameters) == self.__len__())
        weights_dof = self._nn.weights[walker_index].get_total_dof()
        self._nn.weights[walker_index].assign(self._nn.sess, parameters[0:weights_dof])
        self._nn.biases[walker_index].assign(self._nn.sess, parameters[weights_dof:])

    def __repr__(self):
        """ Prints all parameters of all walkers.

        :return: string with all parameters from all walkers
        """
        output = "["
        for i in range(self.num_walkers()):
            if i != 0:
                output += ","
            output += str(self.__getitem__(i))
        output += "]"
        return output