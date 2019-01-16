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
import time

from TATi.model import Model
from TATi.options.commandlineoptions import CommandlineOptions
from TATi.runtime.runtime import runtime

options = CommandlineOptions()


def parse_parameters():
    """ Sets up the argument parser for parsing command line parameters into dictionary

    :return: dictionary with parameter names as keys, unrecognized parameters
    """
    global options

    options.add_common_options_to_parser()
    options.add_data_options_to_parser()
    options.add_model_options_to_parser()
    options.add_prior_options_to_parser()
    options.add_train_options_to_parser()

    return options.parse()

def main(_):
    global options

    rt = runtime(options)

    time_zero = time.process_time()

    network_model = Model(options)

    time_init_network_zero = time.process_time()

    network_model.init_input_pipeline()
    network_model.init_network(options.restore_model, setup="train")
    network_model.reset_dataset()

    rt.set_init_network_time(time.process_time() - time_init_network_zero)

    network_model.train()

    if options.do_hessians:
        network_model.compute_optimal_stepwidth()

    rt.set_train_network_time(time.process_time() - rt.time_init_network)

    rt.set_overall_time(time.process_time() - time_zero)

def internal_main():
    global options

    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    unparsed = parse_parameters()

    options.react_to_common_options()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

