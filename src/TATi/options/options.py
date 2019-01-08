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

    def __contains__(self, key):
        """ Checks whether 'key` is a known key to this options class.

        :param key: key name
        :return: True - key found, False - key unknown
        """
        return str(key) in self._option_map.keys()

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
        if key in self._option_map.keys():
            if key in self._type_map.keys():
                if (value is None) or isinstance(value, self._type_map[key]):
                    self._option_map[key] = value
                else:
                    raise ValueError("Option %s needs to be of type %s"  \
                                     % (key,str(self._type_map[key])))
            elif key in self._list_type_map.keys():
                if isinstance(value, list):
                    for i in value:
                        if not isinstance(i, self._list_type_map[key]):
                            raise ValueError(("Value %s in option %s needs to be "+ \
                                             "list of type %s") % (str(i), key, str(self._list_type_map[key])))
                    if isinstance(self._option_map[key], list):
                        self._option_map[key][:] = value
                    else:
                        self._option_map[key] = value
                else:
                    raise ValueError("Option %s needs to be list of type %s" \
                                     %  (key, str(self._list_type_map[key])))
            else:
                # option has no designated type
                self._option_map[key] = value
        else:
            raise AttributeError("Option '"+key+"' is unknown")

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