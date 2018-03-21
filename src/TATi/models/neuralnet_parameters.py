import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from TATi.models.basetype import dds_basetype

class neuralnet_parameters:
    """ This class wraps methods to get all parameters of a neural network, i.e.
    weights and biases, as a single long vector. And also, the other way round to
    set all parameters from a single long vector.

    """

    def __init__(self, _list_of_tensors):
        self.parameters = _list_of_tensors
        self.placeholders = self.create_placeholders(_list_of_tensors)
        assert( len(self.parameters) == len(self.placeholders) )

        # create assign all control group
        assigns = []
        for i in range(len(self.parameters)):
            assigns.append(tf.assign(self.parameters[i], self.placeholders[i]))
        self.assign_all_t = control_flow_ops.group(*assigns)

    def create_flat_vector(self):
        """ Creates a zero-filled numpy array of dimension matching parameters

        :return: zero-filled vector of right dimension
        """
        total_dof = self.get_total_dof_from_list(self.parameters)
        logging.info("Number of dof: " + str(total_dof))

        # setup flat np array
        return np.zeros([total_dof])

    def evaluate(self, _sess):
        """ Evaluates the parameters and returns a flat vector.

        :return: flat vector of parameters' current values
        """
        # evaluate all
        weights_eval = _sess.run(self.parameters)
        #logging.debug(weights_eval)

        # convert to flat array
        return self.flatten_list_of_arrays(weights_eval)

    def assign(self, _sess, _np_array):
        """ Assigns all the parameters given a flat vector of values.

        :param _sess: tensorflow session
        """
        # convert to same shape as placeholders/weight tensors
        list_of_arrays = self.convert_np_array_to_match_list_of_tensors(
            _np_array, self.placeholders)

        # assign all
        feed_dict = self.create_feed_dict(self.placeholders, list_of_arrays)
        _sess.run(self.assign_all_t, feed_dict=feed_dict)

    def compare_flat_vectors(self, _vector1, _vector2):
        # check equivalence
        assert( _vector1.size == _vector2.size )
        results = [_vector1[i] == _vector2[i] for i in range(_vector1.size)]
        return all(results)

    def get_total_dof(self):
        return neuralnet_parameters.get_total_dof_from_list(self.parameters)

    @staticmethod
    def create_placeholders(_list_of_tensors):
        """ Create a list of tensors of placeholders given a list of tensors
         of variables, i.e. all match in size

        :param _list_of_tensors: list of tensors of variables
        :return: list of tensors of placeholders with equivalent sizes
        """
        weights_placeholder_list = []
        for tensor in _list_of_tensors:
            weights_placeholder_list.append(
                tf.placeholder(shape=tensor.get_shape(), dtype=dds_basetype)
            )
            logging.debug("Weight: " + str(tensor))
            logging.debug("Placeholder: " + str(weights_placeholder_list[-1]))
        return weights_placeholder_list

    @staticmethod
    def get_dof(_tensor):
        dof = 0
        dims = _tensor.get_shape()
        if len(dims) > 0 and dims[0] != 0:
            tmp_dof = 1
            for j in dims:
                tmp_dof *= int(j)
            dof = tmp_dof
        return dof

    @staticmethod
    def get_total_dof_from_list(_list_of_tensors):
        # get the number of total weights
        return sum([neuralnet_parameters.get_dof(i) for i in _list_of_tensors])

    @staticmethod
    def convert_np_array_to_match_list_of_tensors(
            _numpy_array,
            _list_of_tensors):
        index = 0
        list_of_array = []
        total_dof = 0
        for tensor in _list_of_tensors:
            dof = neuralnet_parameters.get_dof(tensor)
            total_dof += dof
            partial_array = _numpy_array[index:index + dof]
            index += dof
            list_of_array.append(
                np.reshape(partial_array, tensor.get_shape(), order='C')
            )
        assert (total_dof == index)
        return list_of_array

    @staticmethod
    def flatten_list_of_arrays(list_of_array):
        flattened = []
        for array in list_of_array:
            flat_array = np.reshape(array, [array.size])
            flattened.append(flat_array)
        if len(flattened) > 0:
            return np.concatenate(flattened)
        else:
            return np.empty(shape=(0))

    @staticmethod
    def create_feed_dict(list_of_tensors, list_of_arrays):
        feed_dict = dict(zip(list_of_tensors, list_of_arrays))
        return feed_dict



