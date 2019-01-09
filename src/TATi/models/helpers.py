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

from TATi.options.pythonoptions import PythonOptions


def get_dimension_from_tfrecord(filenames):
    """ Helper function to get the size of the dataset contained in a TFRecord.

    :param filenames: list of tfrecord files
    :return: total size of dataset
    """
    dimension = 0
    for filename in filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=filename)
        for string_record in record_iterator:
            if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
                example = tf.train.Example()
                example.ParseFromString(string_record)
                _ = int(example.features.feature['height']
                             .int64_list
                             .value[0])

                _ = int(example.features.feature['width']
                            .int64_list
                            .value[0])
                # logging.debug("height is "+str(height)+" and width is "+str(width))
            dimension += 1

    logging.info("Scanned " + str(dimension) + " records in tfrecord file.")

    return dimension


def setup_parameters(_, **kwargs):
    return PythonOptions(add_keys=True, value_dict=kwargs)

