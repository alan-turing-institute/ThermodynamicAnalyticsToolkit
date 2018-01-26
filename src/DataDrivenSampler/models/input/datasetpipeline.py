import tensorflow as tf

import functools
from math import ceil
import sys

from DataDrivenSampler.common import decode_csv_line, read_and_decode_image, get_csv_defaults
from DataDrivenSampler.models.input.inputpipeline import InputPipeline

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

        defaults = get_csv_defaults(
            input_dimension=input_dimension,
            output_dimension=output_dimension)
        #print(defaults)
        self.dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if filetype == "csv":
            self.dataset = self.dataset.flat_map(
                lambda filename: (
                    tf.data.TextLineDataset(filename)
                        .skip(1)
                        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), '#'))
                        .cache()))
            self.dataset = self.dataset.map(functools.partial(decode_csv_line, defaults=defaults))
        elif filetype == "tfrecord":
            self.dataset = self.dataset.flat_map(
                lambda filename: (tf.data.TFRecordDataset(filename)))
            # TODO: this is very specific at the moment
            self.dataset = self.dataset.map(functools.partial(read_and_decode_image,
                                                              num_pixels=input_dimension,
                                                              num_classes=output_dimension))
        else:
            print("Unknown filetype")
            sys.exit(255)
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.repeat(ceil(max_steps*batch_size/dimension))
        print(self.dataset.output_shapes)
        print(self.dataset.output_types)

        self.iterator = self.dataset.make_initializable_iterator()
        self.batch_next = self.iterator.get_next()

    def next_batch(self, session):
        ''' This returns the next batch of features and labels.

        :param session: session object as input might be retrieved through the
                computational graph
        :return: pack of feature and label array
        '''
        # fetch next batch of data
        try:
            batch_data = session.run(self.batch_next)
        except tf.errors.OutOfRangeError:
            self.reset(session)
            try:
                batch_data = session.run(self.batch_next)
            except tf.errors.OutOfRangeError:
                print('Dataset is too small for one batch!')
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
        print("WARNING: Needing to reset the iterator running over the dataset.")
        session.run(self.iterator.initializer)

    def shuffle(self, seed):
        ''' This shuffles the dataset.

        :warning: this may require to reset() the dataset as well.

        :param seed: random number seed for shuffling to allow reproducible runs
        '''
        self.dataset = self.dataset.shuffle(seed=seed)
