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
    """Helper function to get the size of the dataset contained in a TFRecord.

    Args:
      filenames: list of tfrecord files

    Returns:
      total size of dataset

    """
    dimension = 0
    for filename in filenames:
        record_iterator = tf.python_io.tf_record_iterator(path=filename)
        for string_record in record_iterator:
            if logging.getLogger(__name__).isEnabledFor(logging.DEBUG):
                example = tf.train.Example()
                example.ParseFromString(string_record)
                #height = \
                int(example.features.feature['height']
                             .int64_list
                             .value[0])

                #width = \
                int(example.features.feature['width']
                            .int64_list
                            .value[0])
                # logging.debug("height is "+str(height)+" and width is "+str(width))
            dimension += 1

    logging.info("Scanned " + str(dimension) + " records in tfrecord file.")

    return dimension


def get_weight_and_bias_column_numbers(df_parameters):
    """Returns two lists with all weight and bias numbers retrieved from
    the `df_parameters`'s column names.

    Args:
      df_parameters: dataframe whose column names to inspect

    Returns:
      weight number list and bias number list

    """
    weight_numbers = []
    bias_numbers = []
    for keyname in df_parameters.columns:
        if "0" <= keyname[1] <= "9":
            if "w" == keyname[0]:
                weight_numbers.append(int(keyname[1:]))
            elif "b" == keyname[0]:
                bias_numbers.append(int(keyname[1:]))
        else:
            if "weight" in keyname:
                weight_numbers.append(int(keyname[6:]))
            elif ("bias" in keyname):
                bias_numbers.append(int(keyname[4:]))

    return weight_numbers, bias_numbers


def get_start_index_in_dataframe_columns(numbers, df, paramlist):
    start_index = -1
    if len(numbers) > 0:
        for param in paramlist:
            try:
                start_index = df.columns.get_loc("%s%d" % (param, numbers[0]))
                break
            except KeyError:
                pass
    return start_index


def check_aligned(numbers):
    lastnr = None
    for nr in numbers:
        if lastnr is not None:
            if lastnr >= nr:
                break
        lastnr = nr
    return lastnr


def check_column_names_in_order(df_parameters, weight_numbers, bias_numbers):
    weights_start = get_start_index_in_dataframe_columns(
        weight_numbers, df_parameters, ["weight", "w"])
    biases_start = get_start_index_in_dataframe_columns(
        bias_numbers, df_parameters, ["bias", "b"])

    weights_aligned = (len(weight_numbers) > 0) and (check_aligned(weight_numbers) == weight_numbers[-1])
    biases_aligned = (len(bias_numbers) > 0) and (check_aligned(bias_numbers) == bias_numbers[-1])
    values_aligned = weights_aligned and biases_aligned and weights_start < biases_start
    return values_aligned, weights_aligned, biases_aligned


def setup_parameters(_, **kwargs):
    return PythonOptions(add_keys=True, value_dict=kwargs)

