""" @package docstring
Options contains all parameters that control runtime behavior.

"""

import logging

class Options(object):
    """ Options is an abstract class to contain all option values.

    Example:
        >>> o=Options()
        >>> print(o.step_width)
        AttributeError: Option 'step_width' is unknown.
        >>> o.add("step_width")
        >>> o.step_width = 0.01
        >>> o.add("step_width")
        AttributeError: Option 'step_width' is already known.

    Warning:
        If you add new member variables to this class or any derived
        class, then you need to add its string name to `_excluded_keys`.
        This is necessary as we have overridden `__getattr__` and
        `__setattr__` to allow access to option names using the "dot"
        operator.

    Raises:
        AttributeError (when option name unknown)
    """

    def __init__(self):
        """ Creates the internal option map that contains all option values.
        """
        # every real member variables must be stored in this list
        self._excluded_keys = ["_option_map"]
        # map of all options
        self._option_map = {}

    def __getstate__(self):
        """ Override pickling's `getstate()` as otherwise the `_options_map`
        is searched for the builtin function.



        """
        return {"_option_map": self._option_map,
                "_excluded_keys": self._excluded_keys
                }

    def __setstate__(self, state):
        """ Override pickling's `setstate()` as otherwise the `_options_map`
        is search for the builtin function.
        """
        self._option_map = state["_option_map"]
        self._excluded_keys = state["_excluded_keys"]

    def add(self, key):
        """ Specifically add a new value to the set of options.

        @note `set()` will fail when the option is not known, yet. This is done
        deliberately to prevent errors with typos in option names.

        @note Option is initially set to None.

        Raises:
            AttributeError

        :param key: name of new option
        """
        if key in self._option_map.keys():
            raise AttributeError("Option "+str(key)+" is already known.")
        if key[0] == '_':
            raise AttributeError("Option names as "+str(key)+" with initial underscores are invalid.")
        self._option_map[key] = None

    def get(self, key):
        """ Getter for the value associated with the option named `key`.

        Raises:
            AttributeError

        :param key: name of option
        :return: option value of key
        """
        try:
            return self._option_map[str(key)]
        except KeyError:
            raise AttributeError("Option '"+str(key)+"' is unknown")

    def __getattr__(self, key):
        """ Override `__getattr__` to mask access to options as if they were
        member variables.

        Raises:
            AttributeError

        :param key: name of option
        :return: value of option
        """
        #print("__getattr__:"+key)
        if (key[0] != '_') and \
                (key != "_excluded_keys") and (key not in self._excluded_keys):
            return self.get(key)
        return super(Options, self).__getattribute__(key)

    def set(self, key, value):
        """ Setter for the value associated with the option named `key`.

        Raises:
            AttributeError

        :param key: name of option
        :param value: value to set
        """
        if str(key) in self._option_map.keys():
            self._option_map[str(key)] = value
        else:
            raise AttributeError("Option '"+str(key)+"' is unknown")

    def __setattr__(self, key, value):
        """ Override `__setattr__` to mask access to options as if they were
        member variables.

        Raises:
            AttributeError

        :param key: name of option
        :param value: value to set
        :return: value of option
        """
        #print("__setattr__:"+key)
        if (key[0] != '_') and \
                (key != "_excluded_keys") and (key not in self._excluded_keys):
            self.set(key, value)
        else:
            super(Options,self).__setattr__(key, value)

    def __repr__(self):
        output="{"
        first_value = True
        for key in sorted(self._option_map.keys()):
            if not first_value:
                output += ", "
            else:
                first_value=False
            output += "'"+key+"': "+str(self._option_map[key])
        output += "}"
        return output