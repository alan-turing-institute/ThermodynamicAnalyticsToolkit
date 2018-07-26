class EvaluationCache(object):
    """ This class caches tensorflow evaluation with respect to certain nodes
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

    """
    def __init__(self, nn):
        self._nn = nn
        self._cache = {}
        self._node_keys = {}
        self._features = None
        self._labels = None

    def _init_node_keys(self):
        """ Initializes the set of cached variables with nodes from the tensorflow's
        graph.

        Note:
            As we may initialize first after having set up the network, this
            needs to go into an extra function.
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
        """ Getter to the current dataset.

        :return: features, labels
        """
        if self._features is None or self._labels is None:
            self._update_cache()
        return self._features, self._labels

    def reset(self):
        """ This resets the cache to None for all cached variables.
        """
        for key in self._node_keys.keys():
            self._cache[key] = [None]*self._nn.FLAGS.number_walkers
        self._features = None
        self._labels = None

    def _return_cache(self, key, walker_index=None):
        """ This returns the cached element named `key` and resets the entry.

        :param key: key of cached variable
        :param walker_index: index of walker to access or None for all
        :return: value(s) stored in cache for `key`
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
        """ Checks whether a value to `key` is in cache at the moment

        :param key: name to cached variable
        :param walker_index: index of walker to access or None for all
        :return: True - values are cached, False - None is stored in cache
        """
        if walker_index is None:
            return all([a is not None for a in self._cache[key]])
        else:
            return self._cache[key][walker_index] is not None

    def _update_cache(self):
        """ Updates the contents of the cache from the network's values.
        """
        self._advance_iterator()
        nodes = list(self._node_keys.values())
        values = self._evaluate(nodes)
        for key, value in zip(self._node_keys.keys(), values):
            self._cache[key] = value

    def _advance_iterator(self):
        """ Advances the iterator and caches dataset internally.
        """
        self._features, self._labels = self._nn.input_pipeline.next_batch(
            self._nn.sess, auto_reset=True)

    def _evaluate(self, nodes):
        """ Helper function to evaluate an arbitrary node of tensorflow's
        internal graph.

        :param nodes: node to evaluate in `session.run()`
        :return: result of node evaluation
        """
        feed_dict = {
            self._nn.xinput: self._features,
            self._nn.nn[0].placeholder_nodes["y_"]: self._labels}
        eval_nodes = self._nn.sess.run(nodes, feed_dict=feed_dict)
        return eval_nodes


    def evaluate(self, key, walker_index=None):
        """ Evaluates the variable `key` either from cache or updating if not
        present there.

        :param key: key of variable
        :param walker_index: index of walker to access or None for all
        :return: value to current batch of `key`
        """
        if not self._check_cache(key, walker_index):
            self._update_cache()
        return self._return_cache(key, walker_index)

    def hasNode(self, key):
        """ Returns whether `key` is among the cached values.

        :param key: name of node
        :return: True - is cache, False - not cached
        """
        return key in self._node_keys.keys()