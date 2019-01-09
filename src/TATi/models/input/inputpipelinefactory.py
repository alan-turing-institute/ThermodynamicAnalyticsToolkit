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

from TATi.common import file_length
from TATi.models.input.datasetpipeline import DatasetPipeline
from TATi.models.input.inmemorypipeline import InMemoryPipeline
from TATi.models.helpers import get_dimension_from_tfrecord


class InputPipelineFactory(object):
    """ This factory produces specialized instances of `InputPipeline`.

    """
    @staticmethod
    def create(FLAGS, shuffle=False):
        """ This creates an input pipeline using the tf.Dataset module.

        :param FLAGS: parameters
        :param shuffle: whether to shuffle dataset or not
        """
        InputPipelineFactory.check_valid_batch_size(FLAGS)
        if FLAGS.in_memory_pipeline:
            logging.debug("Using in-memory pipeline")
            # create a session, parse the tfrecords with batch_size equal to dimension
            input_pipeline = DatasetPipeline(
                filenames=FLAGS.batch_data_files, filetype=FLAGS.batch_data_file_type,
                batch_size=FLAGS.dimension, dimension=FLAGS.dimension, max_steps=1,
                input_dimension=FLAGS.input_dimension, output_dimension=FLAGS.output_dimension,
                shuffle=shuffle, seed=FLAGS.seed)
            with tf.Session() as session:
                session.run(input_pipeline.iterator.initializer)
                xs, ys = input_pipeline.next_batch(session)

            input_pipeline = InMemoryPipeline(dataset=[xs, ys], batch_size=FLAGS.batch_size,
                                              max_steps=FLAGS.max_steps,
                                              shuffle=shuffle, seed=FLAGS.seed)
        else:
            logging.debug("Using tf.Dataset pipeline")
            input_pipeline = DatasetPipeline(filenames=FLAGS.batch_data_files, filetype=FLAGS.batch_data_file_type,
                                             batch_size=FLAGS.batch_size, dimension=FLAGS.dimension,
                                             max_steps=FLAGS.max_steps, input_dimension=FLAGS.input_dimension,
                                             output_dimension=FLAGS.output_dimension,
                                             shuffle=shuffle, seed=FLAGS.seed)
        return input_pipeline

    @staticmethod
    def provide_data(FLAGS, features, labels, shuffle=False):
        """ Use to provide an in-memory dataset, i.e., numpy arrays with
        `features` and `labels`.

        :param features: feature part of dataset
        :param labels: label part of dataset
        :param shuffle: whether to shuffle the dataset initially or not
        """
        logging.info("Using in-memory pipeline")
        FLAGS.input_dimension = len(features[0])
        FLAGS.output_dimension = len(labels[0])
        try:
            FLAGS.add("output_type")
        except AttributeError:
            # add only on first call
            pass
        if FLAGS.output_dimension == 1:
            FLAGS.output_type = "binary_classification"  # labels in {-1,1}
        else:
            FLAGS.output_type = "onehot_multi_classification"
        assert(len(features) == len(labels))
        try:
            FLAGS.dimension
        except AttributeError:
            FLAGS.add("dimension")
        FLAGS.dimension = len(features)
        InputPipelineFactory.check_valid_batch_size(FLAGS)
        input_pipeline = InMemoryPipeline(dataset=[features, labels],
                                          batch_size=FLAGS.batch_size,
                                          max_steps=FLAGS.max_steps,
                                          shuffle=shuffle, seed=FLAGS.seed)
        return input_pipeline


    @staticmethod
    def check_valid_batch_size(FLAGS):
        ''' Helper function to check that batch_size does not exceed dimension
        of dataset. After which it will be valid.

        :return: True - is smaller or equal, False - exceeded and capped batch_size
        '''
        if FLAGS.batch_size is None:
            logging.info("batch_size not set, setting to dimension of dataset.")
            FLAGS.batch_size = FLAGS.dimension
            return True
        if FLAGS.batch_size > FLAGS.dimension:
            logging.warning(" batch_size exceeds number of data items, capping.")
            FLAGS.batch_size = FLAGS.dimension
            return False
        else:
            return True


    @staticmethod
    def scan_dataset_dimension_from_files(FLAGS):
        input_dimension, output_dimension = None, None
        if len(FLAGS.batch_data_files) > 0:
            input_dimension = FLAGS.input_dimension
            output_dimension = FLAGS.output_dimension
            try:
                FLAGS.add("dimension")
            except AttributeError:
                # add only on first call
                pass
            try:
                FLAGS.add("output_type")
            except AttributeError:
                # add only on first call
                pass
            if FLAGS.batch_data_file_type == "csv":
                FLAGS.dimension = sum([file_length(filename)
                                            for filename in FLAGS.batch_data_files]) \
                                       - len(FLAGS.batch_data_files)
                if output_dimension == 1:
                    FLAGS.output_type = "binary_classification"  # labels in {-1,1}
                else:
                    FLAGS.output_type = "onehot_multi_classification"
            elif FLAGS.batch_data_file_type == "tfrecord":
                FLAGS.dimension = get_dimension_from_tfrecord(FLAGS.batch_data_files)
                FLAGS.output_type = "onehot_multi_classification"
            else:
                logging.info("Unknown file type")
                assert(0)

            logging.info("Parsing "+str(FLAGS.batch_data_files))

        return input_dimension, output_dimension
