import functools
import logging
from math import ceil
import sys
import tensorflow as tf

from TATi.common import decode_csv_line, read_and_decode_image, get_csv_defaults
from TATi.models.input.inputpipeline import InputPipeline

class DatasetPipeline(InputPipeline):
    """ This specific input pipeline uses the tf.Dataset module to parse CSV
    (or other) data from file and present it to the network in batches.

    """

    def __init__(self, filenames, filetype,
                 batch_size, dimension, max_steps,
                 input_dimension, output_dimension,
                 shuffle, seed):
        ''' Initializes the tf.Dataset object by supplying CSV filenames, decoding
        them, and putting them into batches.

        :param filenames: list of filenames to parse
        :param filetype: type of the files to parse: csv, tfrecord
        :param batch_size: number of datums to return
        :param dimension: number of datums in total
        :param max_steps: maximum number of steps
        :param input_dimension: number of nodes in the input layer/number of features
        :param output_dimension: number of nodes in the output layer/number of labels
        :param shuffle: whether to shuffle dataset initially or not
        :param seed: seed used for random shuffle to allow reproducible runs
        '''

        self.dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if filetype == "csv":
            defaults = get_csv_defaults(
                input_dimension=input_dimension,
                output_dimension=output_dimension)
            logging.debug(defaults)
            self.dataset = self.dataset.flat_map(
                lambda filename: (
                    tf.data.TextLineDataset(filename)
                        .skip(1)
                        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), '#'))
                        .cache()))
            self.dataset = self.dataset.map(functools.partial(decode_csv_line, defaults=defaults,
                                                              input_dimension=input_dimension,
                                                              output_dimension=output_dimension))
        elif filetype == "tfrecord":
            self.dataset = self.dataset.flat_map(
                lambda filename: (tf.data.TFRecordDataset(filename)))
            # TODO: this is very specific at the moment
            self.dataset = self.dataset.map(functools.partial(read_and_decode_image,
                                                              num_pixels=input_dimension,
                                                              num_classes=output_dimension))
        else:
            logging.critical("Unknown filetype")
            sys.exit(255)
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=10*batch_size, seed=seed)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.repeat(ceil(max_steps*batch_size/dimension))
        #logging.info(self.dataset.output_shapes)
        #logging.info(self.dataset.output_types)

        self.iterator = self.dataset.make_initializable_iterator()
        self.batch_next = self.iterator.get_next()

    def next_batch(self, session, auto_reset = False, warn_when_reset = False):
        ''' This returns the next batch of features and labels.

        :param session: session object as input might be retrieved through the
                computational graph
        :param auto_reset: whether to automatically reset the dataset iterator
                or whether the exception tf.errors.OutOfRangeError is not caught
        :param warn_when_reset: whether to warn when reset, requires makes
                auto_reset set to True
        :return: pack of feature and label array
        '''
        # fetch next batch of data
        assert( (not warn_when_reset) or (auto_reset and warn_when_reset) ) # make sure both are activated
        if not auto_reset:
            batch_data = session.run(self.batch_next)
        else:
            try:
                batch_data = session.run(self.batch_next)
            except tf.errors.OutOfRangeError:
                if warn_when_reset:
                    logging.warning("Need to reset the dataset iterator, intended?")
                self.reset(session)
                try:
                    batch_data = session.run(self.batch_next)
                except tf.errors.OutOfRangeError:
                    logging.critical('Dataset is too small for one batch!')
                    sys.exit(255)
        return batch_data[0], batch_data[1]

    def epochFinished(self):
        ''' This checks whether the epoch is done, i.e. whether the dataset
        is depleted and needs to be reset.

        :return: True - epoch is done, False - else
        '''
        pass

    def reset(self, session):
        ''' This resets the dataset such that a new epoch of training or
        sampling may commence.

        :param session: session object as input might be retrieved through the
                computational graph
        '''
        session.run(self.iterator.initializer)

    def shuffle(self, seed):
        ''' This shuffles the dataset.

        :warning: this may require to reset() the dataset as well.

        :param seed: random number seed for shuffling to allow reproducible runs
        '''
        self.dataset = self.dataset.shuffle(seed=seed)
