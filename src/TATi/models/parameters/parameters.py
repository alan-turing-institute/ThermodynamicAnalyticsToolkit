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

import itertools

class Parameters(object):
    """This class overrides `__getitem__` and `__setitem__` to allow simple
    access to the parameters of each walker.

    Args:

    Returns:

    """

    def __init__(self, nn, param_names, _cache=None):
        """

        Args:
          nn: ref to model object
          param_names: names of parameters to represent inside model, e.g. ["weights", "biases"]
          _cache: the cache is notified of updated parameters on `__setitem` (Default value = None)

        Returns:

        """
        self._nn = nn
        if len(param_names) != 2:
            raise ValueError("We need two parameter names, one for weights, one for biases.")
        self._param_names = param_names
        self._cache = _cache

    def __len__(self):
        """Returns the number of walkers, i.e. the number of parameter sets
        of same length.

        Args:

        Returns:
          number of parameters

        """
        return max(1,self._nn.FLAGS.number_walkers)

    def num_parameters(self, walker_index=0):
        """Returns the number of parameters.

        Args:
          walker_index: index of walker (Default value = 0)

        Returns:
          number of walkers/parameter sets

        """
        self._valid_walker_index(walker_index)
        return getattr(self._nn, self._param_names[0])[walker_index].get_total_dof() \
               + getattr(self._nn, self._param_names[1])[walker_index].get_total_dof()

    def _valid_walker_index(self, walker_index):
        """Ascertains that `walker_index` is in valid range.

        Args:
          walker_index: index of walker

        Returns:
          True - walker index is valid, False - not

        """
        assert( 0 <= walker_index < self.__len__() )

    def __getitem__(self, walker_index):
        """Returns the parameters of a specific walker `walker_index`.

        Args:
          walker_index: Index of the walker

        Returns:
          parameter set of the respective walker

        """
        self._valid_walker_index(walker_index)
        try:
            weights_eval = \
                getattr(self._nn, self._param_names[0])[walker_index].evaluate(self._nn.sess)
            biases_eval = \
                getattr(self._nn, self._param_names[1])[walker_index].evaluate(self._nn.sess)
            return list(itertools.chain(weights_eval, biases_eval))
        except AttributeError:
            raise ValueError("The current sampler does not have momenta.")

    def __setitem__(self, walker_index, parameters):
        """Sets the parameters for a specific walker

        Args:
          walker_index: Index of the walker
          parameters: set of new parameters

        Returns:

        """
        self._valid_walker_index(walker_index)
        assert(len(parameters) == self.num_parameters())
        weights_dof = getattr(self._nn, self._param_names[0])[walker_index].get_total_dof()
        try:
            getattr(self._nn, self._param_names[0])[walker_index].assign(
                self._nn.sess, parameters[0:weights_dof])
            getattr(self._nn, self._param_names[1])[walker_index].assign(
                self._nn.sess, parameters[weights_dof:])
        except AttributeError:
            raise ValueError("The current sampler does not have momenta.")
        # tell evaluation cache that parameters have changed
        if self._cache is not None:
            self._cache.invalidate_cache(walker_index)

    def __repr__(self):
        """Prints all parameters of all walkers.

        Args:

        Returns:
          string with all parameters from all walkers

        """
        output = "["
        for i in range(self.__len__()):
            if i != 0:
                output += ","
            output += str(self.__getitem__(i))
        output += "]"
        return output