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
import sys
import tensorflow as tf

from TATi.common import get_list_from_string
from TATi.model import Model
from TATi.models.input.inputpipelinefactory import InputPipelineFactory
from TATi.options.commandlineoptions import CommandlineOptions
from TATi.samplers.grid.samplingmodes import SamplingModes

output_width=8
output_precision=8

def get_description():
    output = "The 'mode' parameter determines which way the sampling is executed.\n"
    output += "It switches between grid-based sampling and resampling a given"+ \
        " trajectory.\n"
    output += "'trajectory_file' contains a trajectory from a sampling or "+ \
        " optimization run. Either a certain step, defined by 'trajectory_stepnr'"+ \
        " is used to define the grid's center or the entire trajectory is taken"+ \
        " for resampling.\n The latter may be used to switch the dataset (e.g., "+ \
        " test instead of training) or to change the batch size (no "+ \
        " mini-batching), i.e. to recalculate the loss for a given trajectory.\n"
    output += "In case the output is desired only on a subspace of the original"+ \
        " parameter space, then a 'directions_file' needs to be specified. Its"+ \
        " rows each define a vector spanning said subspace.\n"
    return output

options = CommandlineOptions(description=get_description())

def parse_parameters():
    """ Sets up the argument parser for parsing command line parameters into dictionary

    :return: dictionary with parameter names as keys, unrecognized parameters
    """
    global options

    options._add_option_cmd('mode', type=str, default=None,
        help='sampling mode to execute: '+", ".join(SamplingModes.list_modes()))

    options.add_data_options_to_parser()
    options.add_model_options_to_parser()

    options._add_option_cmd('--csv_file', type=str, default=None,
        help='CSV file name to output sampled values to.')
    options._add_option_cmd('--interval_biases', type=str, nargs='+', default=[],
        help='Min and max value for each bias.')
    options._add_option_cmd('--exclude_parameters', type=str, nargs='+', default=[],
        help='List of biases, e.g. b0, and weights, e.g. w3, to exclude from sampling.')
    options._add_option_cmd('--inter_ops_threads', type=int, default=1,
        help='Sets the number of threads to split up ops in between. NOTE: This hurts reproducibility to some extent because of parallelism.')
    options._add_option_cmd('--interval_weights', type=float, nargs='+', default=[],
        help='Min and max value for each weight.')
    options._add_option_cmd('--interval_offsets', type=float, nargs='+', default=[],
        help='Offset to shift centers of intervals per axis')
    options._add_option_cmd('--intra_ops_threads', type=int, default=None,
        help='Sets the number of threads to use within an op, i.e. Eigen threads for linear algebra routines.')
    options._add_option_cmd('--restore_model', type=str, default=None,
        help='Restore model (weights and biases) from a file.')
    options._add_option_cmd('--samples_biases', type=int, default=None,
        help='Number of samples to take per bias interval')
    options._add_option_cmd('--samples_weights', type=int, default=None,
        help='Number of samples to take per weight interval')
    options._add_option_cmd('--verbose', '-v', action='count',
        help='Level of verbosity during compare')
    options._add_option_cmd('--version', '-V', action="store_true",
        help='Gives version information')

    return options.parse()


def main(_):
    global options

    # add options which model requires
    options.add("number_walkers")
    options.number_walkers = 1
    options.add("max_steps")
    options.max_steps = 1

    network_model = Model(options)
    InputPipelineFactory.scan_dataset_dimension_from_files(options)
    network_model.init_network(options.restore_model, setup=None)

    options.exclude_parameters = get_list_from_string(options.exclude_parameters)

    # prepare which mode to use
    sampler = SamplingModes.create(options.mode, network_model, options)
    if sampler is None:
        logging.critical("The mode name "+options.mode+" is not known.")
        sys.exit(255)

    options.max_steps = sampler.get_max_steps()

    print("There are "+str(options.max_steps)+" points to sample.")
    network_model.reset_parameters(options)

    network_model.init_input_pipeline()
    network_model.reset_dataset()
    print(options)

    csv_writer, csv_file = sampler.prepare_csv_output(options.csv_file)

    sampler.goto_start()

    for i in range(options.max_steps):
        coords_eval = sampler.set_step()
        loss_eval, acc_eval = sampler.evaluate_loss()
        if csv_writer is not None:
            sampler.write_output_line(csv_writer, loss_eval, acc_eval, coords_eval,
                                      output_precision, output_width)
        sampler.goto_next_step()

    if csv_file is not None:
        csv_file.close()

def internal_main():
    global options

    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    unparsed = parse_parameters()

    print(options)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

