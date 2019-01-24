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

import functools
import logging
from math import ceil
import sys
import tensorflow as tf

from TATi.common import decode_csv_line, read_and_decode_image, get_csv_defaults
from TATi.models.input.inputpipeline import InputPipeline

class DatasetPipeline(InputPipeline):
    """This specific input pipeline uses the tf.Dataset module to parse CSV
    (or other) data from file and present it to the network in batches.

    Args:

    Returns:

    """

    def __init__(self, filenames, filetype,
                 batch_size, dimension, max_steps,
                 input_dimension, output_dimension,
                 shuffle, seed):
        """Initializes the tf.Dataset object by supplying CSV filenames, decoding
        them, and putting them into batches.

        Args:
          filenames: list of filenames to parse
          filetype: type of the files to parse: csv, tfrecord
          batch_size: number of datums to return
          dimension: number of datums in total
          max_steps: maximum number of steps
          input_dimension: number of nodes in the input layer/number of features
          output_dimension: number of nodes in the output layer/number of labels
          shuffle: whether to shuffle dataset initially or not
          seed: seed used for random shuffle to allow reproducible runs

        Returns:

        """

        self.dataset = tf.data.Dataset.from_tensor_slices(filenames)
        if filetype == "csv":
            defaults = get_csv_defaults(
                input_dimension=input_dimension,
                output_dimension=output_dimension)
            logging.debug(defaults)
            self.dataset = self.dataset.interleave(
                lambda filename: (
                    tf.data.TextLineDataset(filename)
                        .skip(1)
                        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), '#'))),
                cycle_length=self.NUM_PARALLEL_CALLS, block_length=16)
            self.dataset = self.dataset.map(
                functools.partial(decode_csv_line, defaults=defaults,
                                  input_dimension=input_dimension,
                                  output_dimension=output_dimension),
                num_parallel_calls=self.NUM_PARALLEL_CALLS).cache()
        elif filetype == "tfrecord":
            self.dataset = self.dataset.interleave(
                lambda filename: (tf.data.TFRecordDataset(filename)),
                cycle_length=self.NUM_PARALLEL_CALLS, block_length=16)
            # TODO: this is very specific at the moment
            self.dataset = self.dataset.map(
                functools.partial(read_and_decode_image,
                                  num_pixels=input_dimension,
                                  num_classes=output_dimension),
                num_parallel_calls=self.NUM_PARALLEL_CALLS).cache()
        else:
            logging.critical("Unknown filetype")
            sys.exit(255)
        if shuffle:
            self.dataset = self.dataset.shuffle(buffer_size=100*batch_size, seed=seed)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.repeat(ceil(max_steps*batch_size/dimension))
        self.dataset = self.dataset.prefetch(100 * batch_size)
        #logging.info(self.dataset.output_shapes)
        #logging.info(self.dataset.output_types)

        self.iterator = self.dataset.make_initializable_iterator()
        self.batch_next = self.iterator.get_next()

    def next_batch(self, session, auto_reset = False, warn_when_reset = False):
        """This returns the next batch of features and labels.

        Args:
          session: session object as input might be retrieved through the
        computational graph
          auto_reset: whether to automatically reset the dataset iterator
        or whether the exception tf.errors.OutOfRangeError is not caught (Default value = False)
          warn_when_reset: whether to warn when reset, requires makes
        auto_reset set to True (Default value = False)

        Returns:
          pack of feature and label array

        """
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
        """This checks whether the epoch is done, i.e. whether the dataset
        is depleted and needs to be reset.
        
        Args:

        Returns:
          True - epoch is done, False - else

        """
        pass

    def reset(self, session):
        """This resets the dataset such that a new epoch of training or
        sampling may commence.

        Args:
          session: session object as input might be retrieved through the
        computational graph

        Returns:

        """
        session.run(self.iterator.initializer)

    def shuffle(self, seed):
        """This shuffles the dataset.
        
        :warning: this may require to reset() the dataset as well.

        Args:
          seed: random number seed for shuffling to allow reproducible runs

        Returns:

        """
        self.dataset = self.dataset.shuffle(seed=seed)
