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

class EvaluationCache(object):
    """This class caches tensorflow evaluation with respect to certain nodes
    that are interesting in the context of the `simulation` interface.
    
    The reason is that we do not give an explicit function to advance to the
    next dataset batch. Therefore, two subsequent evaluations of the loss
    will advance iterator but also evaluation of loss and gradients.
    
    This caching allows to combine evaluation in an "eager execution"-manner.
    We always evaluate all interesting nodes and cache this result. Accesses
    will first go the the cache but reset the value in their. If we encounter
    a single reset value, the whole cache contents is updated and thereby
    we also step on to the next batch.
    
    To this end, only two functions are important: `evaluate()` returns the
    value for a given key, `reset()` informs the class to the reset its cache
    for external reasons.
    
    Example:
        nn = tati()
        e = EvaluationCache(nn)
        e.evaluate("loss")
        e.reset()

    Args:

    Returns:

    """
    def __init__(self, nn):
        self._nn = nn
        self._cache = {}
        self._node_keys = {}
        self._features = None
        self._labels = None

    def _init_node_keys(self):
        """Initializes the set of cached variables with nodes from the tensorflow's
        graph.
        
        Note:
            As we may initialize first after having set up the network, this
            needs to go into an extra function.

        Args:

        Returns:

        """
        self._node_keys = {
                "loss": self._nn.loss,
                "accuracy": [self._nn.nn[i].get_list_of_nodes(["accuracy"])[0]
                             for i in range(self._nn.FLAGS.number_walkers)]
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

    @property
    def dataset(self):
        """Getter to the current dataset.

        Args:

        Returns:
            features, labels

        """
        if self._features is None or self._labels is None:
            self._update_cache()
        return self._features, self._labels

    def invalidate_cache(self, walker_index):
        """This voids the cache entry for a particular walker.

        Args:
          walker_index: index of walker

        Returns:

        """
        logging.debug("Invalidating cache for #"+str(walker_index))
        for key in self._node_keys.keys():
            if len(self._cache[key]) > 0:
                self._cache[key][walker_index] = None

    def reset(self):
        """This resets the cache to None for all cached variables."""
        for key in self._node_keys.keys():
            self._cache[key] = [None]*self._nn.FLAGS.number_walkers
        self._features = None
        self._labels = None

    def _return_cache(self, key, walker_index=None):
        """This returns the cached element named `key` and resets the entry.

        Args:
          key: key of cached variable
          walker_index: index of walker to access or None for all (Default value = None)

        Returns:
          value(s) stored in cache for `key`

        """
        assert( key in self._cache.keys() )
        if walker_index is None:
            value = []
            value[:] = self._cache[key]
            for i in range(self._nn.FLAGS.number_walkers):
                self._cache[key][i] = None
        else:
            value = self._cache[key][walker_index]
            self._cache[key][walker_index] = None
        return value

    def _check_cache(self, key, walker_index=None):
        """Checks whether a value to `key` is in cache at the moment

        Args:
          key: name to cached variable
          walker_index: index of walker to access or None for all (Default value = None)

        Returns:
          True - values are cached, False - None is stored in cache

        """
        if walker_index is None:
            return all([a is not None for a in self._cache[key]])
        else:
            return self._cache[key][walker_index] is not None

    def _update_cache(self):
        """Updates the contents of the cache from the network's values."""
        logging.debug("Updating evaluationcache")
        self._advance_iterator()
        nodes = list(self._node_keys.values())
        values = self._evaluate(nodes)
        for key, value in zip(self._node_keys.keys(), values):
            self._cache[key] = value

    def _advance_iterator(self):
        """Advances the iterator and caches dataset internally."""
        self._features, self._labels = self._nn.input_pipeline.next_batch(
            self._nn.sess, auto_reset=True)

    def _evaluate(self, nodes):
        """Helper function to evaluate an arbitrary node of tensorflow's
        internal graph.

        Args:
          nodes: node to evaluate in `session.run()`

        Returns:
          result of node evaluation

        """
        feed_dict = {
            self._nn.xinput: self._features,
            self._nn.nn[0].placeholder_nodes["y_"]: self._labels}
        eval_nodes = self._nn.sess.run(nodes, feed_dict=feed_dict)
        return eval_nodes


    def evaluate(self, key, walker_index=None):
        """Evaluates the variable `key` either from cache or updating if not
        present there.

        Args:
          key: key of variable
          walker_index: index of walker to access or None for all (Default value = None)

        Returns:
          value to current batch of `key`

        """
        if not self._check_cache(key, walker_index):
            self._update_cache()
        return self._return_cache(key, walker_index)

    def hasNode(self, key):
        """Returns whether `key` is among the cached values.

        Args:
          key: name of node

        Returns:
          True - is cache, False - not cached

        """
        return key in self._node_keys.keys()