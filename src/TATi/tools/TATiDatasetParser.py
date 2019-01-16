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

import argparse
import functools
import logging
import sys
import tensorflow as tf

from TATi.common import decode_csv_line, get_csv_defaults
from TATi.options.commandlineoptions import react_generally_to_options

FLAGS = None

def main(_):
    global FLAGS

    defaults = get_csv_defaults(input_dimension=2)
    dataset = tf.data.Dataset.from_tensor_slices(FLAGS.batch_data_files)
    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)
            .skip(1)
            .filter(lambda line: tf.not_equal(tf.substr(line, 0,1), '#'))))
    dataset = dataset.map(functools.partial(decode_csv_line, defaults=defaults,
                                            input_dimension=2,
                                            output_dimension=1))

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    sess = tf.Session()
    init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    sess.run(init_op)
    sess.run(iterator.initializer)

    try:
        while True:
            features = sess.run(next_element)
            print(str(features[0])+", "+str(features[1]))
    except tf.errors.OutOfRangeError:
        print('Done training, epoch reached')

def internal_main():
    global FLAGS

    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    parser = argparse.ArgumentParser()
    # please adhere to alphabetical ordering
    parser.add_argument('--batch_data_files', type=str, nargs='+', default=[],
        help='filenames of dataset for training formatted as CSV.')
    parser.add_argument('--batch_size', type=int, default=None,
        help='The number of samples used to divide sample set into batches in one training step.')
    parser.add_argument('--seed', type=int, default=None,
        help='Seed to use for random number generators.')
    parser.add_argument('--verbose', '-v', action='count',
        help='Level of verbosity during compare')
    parser.add_argument('--version', action="store_true",
        help='Gives version information')
    FLAGS, unparsed = parser.parse_known_args()

    react_generally_to_options(FLAGS, unparsed)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

