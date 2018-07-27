import collections
import csv
import logging
import os
import tensorflow as tf

from TATi.models.basetype import dds_basetype

def get_filename_from_fullpath(fullpath):
    """ Returns the filename for any given full path

    :param fullpath: string containing filename and folders
    :return: just the filename
    """
    return os.path.basename(fullpath)

def get_list_from_string(str_or_list_of_str):
    """ Extracts list of strings from any string (or list of strings).

    :param str_or_list_of_str: string
    :return: list of str
    """
    tmpstr=str_or_list_of_str
    if str_or_list_of_str is not str:
        try:
            tmpstr=" ".join(str_or_list_of_str)
        except(TypeError):
            tmpstr=" ".join([item for sublist in str_or_list_of_str for item in sublist])
    return [item for item in tmpstr.split()]


def initialize_config_map():
    """ This initialize the configuration dictionary with default values

    :return:
    """
    # output files
    config_map = {
        "do_write_averages_file": False,
        "do_write_run_file": False,
        "averages_file": None,
        "csv_file": None,
        "do_write_trajectory_file": False,
        "trajectory_file": None
    }

    return config_map


def setup_csv_file(filename, header):
    """ Opens a new CSV file and writes the given `header` to it.

    :param filename: filename of CSV file
    :param header: header to write as first row
    :return: csv writer, csv file
    """
    csv_file = open(filename, 'w', newline='')
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(header)
    return csv_writer, csv_file


def setup_run_file(filename, header, config_map):
    """ Opens the run CSV file if a proper `filename` is given.

    :param filename: filename of run CSV file or None
    :param header: list of strings as header for each column
    :param config_map: configuration dictionary
    :return: CSV writer or None
    """
    if filename is not None:
        config_map["do_write_run_file"] = True
        csv_writer, config_map["csv_file"] = setup_csv_file(filename, header)
        return csv_writer
    else:
        return None


def get_trajectory_header(no_weights, no_biases):
    """ Returns the header for CSV trajectory file based on the given number
    of weights and biases.

    :param no_weights: number of weights of the network
    :param no_biases: number of biases of the network
    :return: list of strings with column names
    """
    return ['id', 'step', 'loss']\
           + [str("weight")+str(i) for i in range(0,no_weights)]\
           + [str("bias") + str(i) for i in range(0, no_biases)]


def setup_trajectory_file(filename, no_weights, no_biases, config_map):
    """ Opens the trajectory file if a proper `filename` is given.

    :param filename: filename of trajectory file or None
    :param config_map: configuration dictionary
    :return: CSV writer or None
    """
    if filename is not None:
        config_map["do_write_trajectory_file"] = True
        trajectory_writer, config_map["trajectory_file"] = \
            setup_csv_file(filename, get_trajectory_header(no_weights, no_biases))
        return trajectory_writer
    else:
        return None


def file_length(filename):
    """ Determines the length of the file designated by `filename`.

    :param filename: name of file
    :return: length
    """
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def number_lines_in_file(filename):
    """ Determines the lines in the file designated by `filename`.

    :param filename: name of file
    :return: number of, lines
    """
    with open(filename) as f:
        for i, l in enumerate(f):
            pass
    return l


def read_from_csv(filename_queue):
    """ Reads a set of records/data from a CSV file into a tensorflow tensor.

    :param filename_queue: filename
    :return: features and labels (i.e. x,y)
    """
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[0.], [0.], [0]]
    col_x1, col_x2, col_label = tf.decode_csv(
        csv_row, record_defaults=record_defaults)
    features = tf.stack([col_x1, col_x2])
    label = tf.stack([col_label])
    return features, label


def read_and_decode_image(serialized_example, num_pixels, num_classes):
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape(num_pixels)

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 tensor.
  label = tf.cast(features['label'], tf.int32)
  label = tf.one_hot(label, num_classes)

  return image, label


def decode_csv_line(line, defaults, input_dimension, output_dimension):
    """Convert a csv line into a (features_dict,label) pair."""
    # Decode the line to a tuple of items based on the types of
    # csv_header.values().
    items = tf.decode_csv(line, list(defaults.values()))

    # reshape into proper tensors
    features = tf.stack(items[0:input_dimension])
    label = tf.reshape(tf.convert_to_tensor(items[input_dimension:input_dimension+output_dimension], dtype=tf.int32), [output_dimension])

    # return last element as label, rest as features
    return features, label


def get_csv_defaults(input_dimension, output_dimension=1):
    """ Return the defaults for a csv line with input features and output labels.

    :param input_dimension: number of features
    :param output_dimension: number of labels
    """
    defaults = collections.OrderedDict([])
    for i in range(input_dimension):
        defaults.update({"x"+str(i+1): [0.]})
    if output_dimension > 1:
        for i in range(output_dimension):
            defaults.update({"label"+str(i+1): [0]})
    else:
        defaults.update({"label": [0]})
    return defaults


def create_input_layer(input_dimension, input_list):
    """ Creates the input layer of TensorFlow's neural network.

    As the input nodes are directly connected to the type of data we feed
     into the network, the function is associated with the dataset generator
     class.

    For arbitrary input dimension we support taking powers, sine or cosine
     of the argument.

    All data resides in the domain [-r,r]^2.

    :param input_dimension: number of nodes for the input layer
    :param input_list: Pick of derived arguments to
            actually feed into the net
    :returns: generated nodes for direct input and derived input
    """
    # Input placeholders
    with tf.name_scope('input'):
        xinput = tf.placeholder(dds_basetype, [None, input_dimension], name='x-input')
        logging.info("xinput is "+str(xinput.get_shape()))

        # parse input columns
        picked_list = []
        for token in input_list:
            # get the argument
            logging.debug("token is " + str(token))
            x_index = token.find('x')
            if x_index != -1:
                arg_name = None
                for i in range(x_index+1, len(token)):
                    if (token[i] < "0") or (token[i] > "9"):
                        arg_name = token[x_index:i]
                        break
                if arg_name is None:
                    arg_name = token[x_index:]
                logging.debug("arg_name is "+str(arg_name))
                arg = xinput[:, (int(arg_name[1:])-1)]
                logging.debug("arg is "+str(arg))
                if "sin" in token:
                    picked_list.append(tf.sin(arg))
                elif "cos" in token:
                    picked_list.append(tf.cos(arg))
                elif "^" in token:
                    power = int(token[(token.find('^')+1):])
                    picked_list.append(tf.pow(arg, power))
                else:
                    picked_list.append(arg)
            else:
                arg = xinput[:, (int(token) - 1)]
                logging.debug("arg is "+str(arg))
                picked_list.append(arg)
        logging.debug("picked_list is "+str(picked_list))
        # if no specific input columns are desired, take all
        if len(input_list) == 0:
            x = tf.identity(xinput)
        else:
            x = tf.transpose(tf.stack(picked_list))
        logging.info("x is " + str(x.get_shape()))
    return xinput, x


def data_numpy_to_csv(dataset, labels, fileName):
    """ Save data in numpy format into .csv file

    params: dataset (numpy.ndarray) with all the points and features
    params: labels (numpy.ndarray) with all the labels
    params: fileName (string) file name with .csv, example: 'filename.csv'
    """
    from TATi.datasets.classificationdatasets \
        import ClassificationDatasets as DatasetGenerator
    import numpy as np

    xs, ys = dataset, labels

    with open(fileName, 'w', newline='') as data_file:
        csv_writer = csv.writer(data_file, delimiter=',', \
                                quotechar='"', \
                                quoting=csv.QUOTE_MINIMAL)
        header = ["x"+str(i+1) for i in range(len(xs[0]))]+["label"]
        csv_writer.writerow(header)
        for x, y in zip(xs, ys):
            csv_writer.writerow(
                ['{:{width}.{precision}e}'.format(val, width=8, precision=8)
    for val in list(x)] \
            + ['{}'.format(y[0], width=8,precision=8)])
    data_file.close()
