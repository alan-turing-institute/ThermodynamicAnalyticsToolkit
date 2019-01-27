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
import math
import numpy as np
import sys
import tensorflow as tf


from TATi.common import setup_csv_file
from TATi.model import Model
from TATi.options.commandlineoptions import CommandlineOptions

options = CommandlineOptions()

output_width=8
output_precision=8


def parse_parameters():
    """ Sets up the argument parser for parsing command line parameters into dictionary

    :return: dictionary with parameter names as keys, unrecognized parameters
    """
    global options

    options.add_data_options_to_parser()
    options.add_model_options_to_parser()

    options._add_option_cmd('--csv_file', type=str, default=None,
        help='CSV file name to output sampled values to.')
    options._add_option_cmd('--inter_ops_threads', type=int, default=1,
        help='Sets the number of threads to split up ops in between. NOTE: This hurts reproducibility to some extent because of parallelism.')
    options._add_option_cmd('--interval_input', type=str, nargs='+', default=[],
        help='Min and max value for each weight.')
    options._add_option_cmd('--intra_ops_threads', type=int, default=None,
        help='Sets the number of threads to use within an op, i.e. Eigen threads for linear algebra routines.')
    options._add_option_cmd('--restore_model', type=str, default=None,
        help='Restore model (input and biases) from a file.')
    options._add_option_cmd('--samples_input', type=int, default=None,
        help='Number of samples to take per weight interval')
    options._add_option_cmd('--verbose', '-v', action='count',
        help='Level of verbosity during compare')
    options._add_option_cmd('--version', '-V', action="store_true",
        help='Gives version information')

    return options.parse()

def main(_):
    global options

    # create the data set
    input_dimension = options.input_dimension

    # sample space
    input_interval_start = float(options.interval_input[0])
    input_interval_end = float(options.interval_input[1])
    input_interval_length = input_interval_end - input_interval_start

    input_linspace = np.arange(0,options.samples_input+1)*input_interval_length/float(
            options.samples_input)+input_interval_start
    input_index_grid = np.zeros(input_dimension, dtype=int)
    input_len_grid = options.samples_input+1

    input_vals = np.zeros(input_dimension, dtype=float)

    input_total_vals = math.pow(options.samples_input+1, input_dimension)

### functions to iterate over input and biases

    def check_end():
        isend = True
        for i in range(input_dimension):
            if input_index_grid[i] != input_len_grid-1:
                isend = False
                break
        return isend

    def next_index():
        for i in range(input_dimension):
            if input_index_grid[i] != input_len_grid-1:
                input_index_grid[i] += 1
                for j in range(i):
                    input_index_grid[j]=0
                break


    ## set up output csv file
    header = [] #["i"]
    for i in range(input_dimension):
        header.append("x"+str(i+1))
    header.append("label")

    csv_writer, csv_file = setup_csv_file(options.batch_data_files[0], header)
    current_step = 0
    is_at_end = False
    while not is_at_end:
        # set the parameters

        input_vals[:] = [input_linspace[ input_index_grid[i] ] for i in range(input_dimension)]

        print_row=[] # [current_step]
        print_row.extend(np.asarray(input_vals))
        print_row.append(0)
        csv_writer.writerow(print_row)

        current_step += 1

        is_at_end = check_end()

        next_index()

    csv_file.flush()
    csv_file.close()

    options.batch_size=1
    options.add("max_steps")
    options.max_steps=input_total_vals
    options.add("number_walkers")
    options.number_walkers=1
    network_model = Model(options)
    network_model.init_input_pipeline()
    network_model.init_network(options.restore_model, setup=None)
    network_model.reset_dataset()

    y = network_model.nn[0].get_list_of_nodes(["y"])
    sess = network_model.sess

    if options.csv_file is not None:
        header = ["i"]
        for i in range(input_dimension):
            header.append("x"+str(i+1))
        header.append("label")
        csv_writer, csv_file = setup_csv_file(options.csv_file, header)

    ## function to evaluate the loss

    def evaluate_label(current_step):
        # get next batch of data
        features, _ = network_model.input_pipeline.next_batch(sess, auto_reset=False)
        print("batch_data is "+str(features))

        # place in feed dict
        feed_dict = {
            network_model.xinput: features,
        }

        print(np.asarray(features[0]))

        label = network_model.sess.run(
            network_model.nn[0].placeholder_nodes["y"],
            feed_dict=feed_dict)

        if options.csv_file is not None:
            print_row = [current_step] \
                        + ['{:{width}.{precision}e}'.format(feature, width=output_width,
                                                            precision=output_precision)
                                                            for feature in np.asarray(features[0])] \
                        + ['{:{width}.{precision}e}'.format(label, width=output_width,
                                                            precision=output_precision)
                                                            for label in np.asarray(label[0])]
            csv_writer.writerow(print_row)

    if options.parse_parameters_file is None:
        print("Need parameters file that contains weights and biases per step.")
        sys.exit(255)
    else:
        for step in options.parse_steps:
            network_model.assign_weights_and_biases_from_file(options.parse_parameters_file, step, do_check=True)

            current_step = 0
            while True:
                try:
                    evaluate_label(current_step)
                    current_step += 1
                except tf.errors.OutOfRangeError:
                    print("End of dataset.")
                    break

    if options.csv_file is not None:
        csv_file.close()

def internal_main():

    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    unparsed = parse_parameters()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

