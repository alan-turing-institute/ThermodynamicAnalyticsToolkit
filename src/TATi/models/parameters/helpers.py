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
import tensorflow as tf


def find_all_in_collections(_collection, _name):
    """ Helper function to return all indices of variables in a collection
     that match with the given `_name`. Note that this removes possible
     walker name scopes.

    :param _collection: collection to search through
    :param _name: tensor/variable name to look for
    :return: list of matching indices
    """
    variable_indices = []
    for i in range(len(_collection)):
        target_name = _collection[i].name
        walker_target_name = target_name[target_name.find("/") + 1:]
        logging.debug("Comparing %s to %s and %s" % (_name, target_name, walker_target_name))
        if target_name == _name or walker_target_name == _name:
            variable_indices.append(i)
    return variable_indices


def extract_from_collections(_collection, _indices):
    """ Helper function to remove all elements associated to each index
    in `indices` from `collections`, gathering them in a list that is
    returned

    :param _collection: collection to remove elements from
    :param _indices: list of indices to extract
    :return: list of elements removed from collection
    """
    variables = []
    _indices.sort(reverse=True)
    for i in _indices:
        variables.append(_collection[i])
        del _collection[i]
    return variables


def fix_parameter_in_collection(_collection, _name, _collection_name="collection"):
    """ Allows to fix a parameter (not modified during optimization
    or sampling) by removing the first instance named _name from trainables.

    :param _collection: (trainables or other) collection to remove parameter from
    :param _name: name of parameter to fix
    :param  _collection_name: name of collection for debugging
    :return: None or Variable ref that was fixed
    """
    variable_indices = find_all_in_collections(_collection, _name)
    logging.debug("Indices matching in " + _collection_name + " with "
                  + _name + ": " + str(variable_indices))
    fixed_variable = extract_from_collections(_collection, variable_indices)
    return fixed_variable


def split_collection_per_walker(_collection, number_walkers):
    """ Helper function to split WEIGHTS and BIASES collection from
    tensorflow into weights and biases per walker.

    :param _collection: collection to split
    :param number_walkers: number of walkers to look for
    :return: list of split up collections
    """
    split_collection = []
    for i in range(number_walkers):
        split_collection.append([])
        scope_name = 'walker'+str(i+1)+'/'
        for var in _collection:
            if scope_name in var.name:
                split_collection[-1].append(var)
    return split_collection


def fix_parameter(_name):
    """ Allows to fix a parameter (not modified during optimization
    or sampling) by removing the first instance named _name from trainables.

    :param _name: name of parameter to fix
    :return: None or Variable ref that was fixed
    """
    trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
    other_collection = None
    other_collection_name = "None"
    if "weight" in _name:
        other_collection = tf.get_collection_ref(tf.GraphKeys.WEIGHTS)
        other_collection_name = "weights"
    elif "bias" in _name:
        other_collection = tf.get_collection_ref(tf.GraphKeys.BIASES)
        other_collection_name = "biases"
    else:
        logging.warning("Unknown parameter category for " + str(_name)
                        + "), removing only from trainables.")

    trainable_variable = fix_parameter_in_collection(trainable_collection, _name, "trainables")
    variable = fix_parameter_in_collection(other_collection, _name, other_collection_name)

    # print(tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES))
    # if "weight" in _name:
    #    print(tf.get_collection_ref(tf.GraphKeys.WEIGHTS))
    # else:
    #    print(tf.get_collection_ref(tf.GraphKeys.BIASES))

    if trainable_variable == variable:
        return variable
    else:
        return None


def assign_parameter(_var, _value):
    """ Creates an assignment node, adding it to the graph.

    :param _var: tensorflow variable ref
    :param _value: value to assign to it, must have same shape
    :return: constant value node and assignment node
    """
    value_t = tf.constant(_value, dtype=_var.dtype)
    assign_t = _var.assign(value_t)
    return value_t, assign_t
