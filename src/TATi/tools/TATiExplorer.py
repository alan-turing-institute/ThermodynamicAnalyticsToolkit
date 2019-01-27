
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
import numpy as np
import tensorflow as tf
import time
import sys

from TATi.common import get_trajectory_header, setup_csv_file
from TATi.exploration.explorer import Explorer
from TATi.exploration.trajectorydatacontainer import TrajectoryDataContainer
from TATi.exploration.trajectoryjobid import TrajectoryJobId
from TATi.model import Model
from TATi.options.commandlineoptions import CommandlineOptions, str2bool
from TATi.runtime.runtime import runtime

from multiprocessing.managers import BaseManager

class MyManager(BaseManager):
    pass

MyManager.register('network_model', Model)
MyManager.register('list', list)
MyManager.register('TrajectoryDataContainer', TrajectoryDataContainer)
MyManager.register('TrajectoryJobId', TrajectoryJobId)

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
    options.add_sampler_options_to_parser()
    options.add_train_options_to_parser()

    # please adhere to alphabetical ordering
    options._add_option_cmd('--cornerpoints_file', type=str, default=None,
        help='Filename to write found corner points to')
    options._add_option_cmd('--diffusion_map_method', type=str, default='vanilla',
        help='Method to use for computing the diffusion map: pydiffmap, vanilla or TMDMap')
    options._add_option_cmd('--max_exploration_steps', type=int, default=2,
        help='Maximum number of exploration steps')
    options._add_option_cmd('--max_legs', type=int, default=100,
        help='Maximum number of legs per trajectory')
    options._add_option_cmd('--minima_file', type=str, default=None,
        help='Filename to write found minima to')
    options._add_option_cmd('--number_of_eigenvalues', type=int, default=4,
        help='How many largest eigenvalues to compute')
    options._add_option_cmd('--number_processes', type=int, default=0,
        help='Whether to activate parallel mode (unequal 0) and how many processes to use then')
    options._add_option_cmd('--number_of_parallel_trajectories', type=int, default=5,
        help='Number of trajectories to run in parallel, i.e. number of points maximally apart in diffusion distance')
    options._add_option_cmd('--number_pruning', type=int, default=0,
        help='Number of pruning stages through metropolis criterion after end of trajectory')
    options._add_option_cmd('--use_reweighting', type=str2bool, default=False,
        help='Use reweighting of the kernel matrix of diffusion maps by the target distribution.')

    return options.parse()


def main(_):
    global options

    rt = runtime(options)
    time_zero = time.process_time()

    if options.number_walkers != 1:
        print("At the moment the exploration code is not adapted to multiple dependent walkers.")
        sys.exit(255)

    # setup neural network
    if options.number_processes == 0:
        network_model = Model(options)
        manager=None
    else:
        manager = MyManager()
        manager.start()
        network_model = manager.network_model(options)

    time_init_network_zero = time.process_time()
    # prepare for both sampling and training
    network_model.init_input_pipeline()
    network_model.init_network(options.restore_model, setup="sample")
    network_model.init_network(options.restore_model, setup="train")
    network_model.reset_dataset()
    rt.set_init_network_time(time.process_time() - time_init_network_zero)

    # prevent writing of output files during leg sampling
    network_model.deactivate_file_writing()

    # 1. run initially just one trajectory
    explorer = Explorer(parameters=options,
                        max_legs=options.max_legs,
                        use_processes=options.number_processes,
                        number_pruning=options.number_pruning,
                        manager=manager)

    # add a uses_ids list
    if options.number_processes == 0:
        used_ids = []
    else:
        used_ids = manager.list()
    explorer.add_used_data_ids_list(used_ids)

    # launch worker processes initially
    if options.number_processes > 0:
        explorer.queue.start_processes(network_model, options)

    print("Creating starting trajectory.")
    # a. add three legs to queue
    explorer.spawn_starting_trajectory(network_model, options.number_of_parallel_trajectories)

    print("Starting multiple explorations from starting trajectory.")
    # 2. with the initial trajectory done and analyzed,
    #    find maximally separate points and sample from these
    cornerpoints = []
    exploration_step = 0
    while exploration_step < options.max_exploration_steps:
        explorer.run_all_jobs(network_model, options)

        # a. combine all trajectories
        steps, parameters, losses = explorer.combine_sampled_trajectories()

        # b. perform diffusion map analysis for eigenvectors
        idx_corner = explorer.get_corner_points(parameters, losses, options, options.number_of_parallel_trajectories)

        # d. spawn new trajectories from these points
        cornerpoints.append( explorer.spawn_corner_trajectories(steps, parameters, losses, idx_corner, network_model) )

        exploration_step += 1

    rt.set_train_network_time(time.process_time() - rt.time_init_network)

    # stop worker processes finally
    if options.number_processes > 0:
        explorer.queue.stop_processes()

    # 3. write final set of values
    data_container = explorer.queue.get_data_container()

    # write away averages time information
    if options.averages_file is not None:
        averages_writer, averages_file = setup_csv_file(options.averages_file, network_model.get_averages_header(setup="sample"))
        for current_id in data_container.get_ids():
            data_object = data_container.get_data(current_id)
            if data_object.type == "sample":
                averages_lines_per_leg = data_object.averages_lines
                for leg_nr in range(len(averages_lines_per_leg)):
                    averages_lines = averages_lines_per_leg[leg_nr]
                    for row in range(len(averages_lines.index)):
                        averages_line = averages_lines.iloc[row,:]
                        averages_line[0] = current_id
                        averages_writer.writerow(averages_line)
        averages_file.close()


    # write away run info
    if options.run_file is not None:
        run_header = network_model.get_run_header(setup="sample")
        run_writer, run_file = setup_csv_file(options.run_file, run_header)
        for current_id in data_container.get_ids():
            data_object = data_container.get_data(current_id)
            if data_object.type == "sample":
                run_lines_per_leg = data_object.run_lines
                for leg_nr in range(len(run_lines_per_leg)):
                    run_lines = run_lines_per_leg[leg_nr]
                    run_lines.iloc[:,0] = current_id
                    for row in range(len(run_lines.index)):
                        run_writer.writerow(run_lines.iloc[row,:])
        run_file.close()

    trajectory_header = get_trajectory_header(
        network_model.get_total_weight_dof(),
        network_model.get_total_bias_dof())

    # write away trajectory information
    if options.trajectory_file is not None:
        trajectory_writer, trajectory_file = \
            setup_csv_file(options.trajectory_file, trajectory_header)
        for current_id in data_container.get_ids():
            data_object = data_container.get_data(current_id)
            if data_object.type == "sample":
                trajectory_lines_per_leg = data_object.trajectory_lines
                for leg_nr in range(len(trajectory_lines_per_leg)):
                    trajectory_lines = trajectory_lines_per_leg[leg_nr]
                    trajectory_lines.iloc[:,0] = current_id
                    for row in range(len(trajectory_lines.index)):
                        trajectory_writer.writerow(trajectory_lines.iloc[row,:])
        trajectory_file.close()

    # write minima to file
    if options.minima_file is not None:
        header = trajectory_header[:]
        header.insert(header.index("loss")+1, "gradient")
        if options.do_hessians:
            insert_index = header.index("gradient")+1
            header.insert(insert_index, "num_negative_ev")
            header.insert(insert_index, "num_positive_ev")
        minima_writer, minima_file = setup_csv_file(
            options.minima_file, header)
        for current_id in data_container.get_ids():
            data_object = data_container.get_data(current_id)
            if data_object.type == "train":
                step = data_object.steps[-1]
                minima_parameters = data_object.parameters[-1]
                loss = data_object.losses[-1]
                gradient = data_object.gradients[-1]
                minima_line = [current_id, step, loss, gradient]
                if options.do_hessians:
                    hessian_ev = data_object.hessian_eigenvalues[-1]
                    num_negative_ev = (0 > hessian_ev).sum()
                    num_positive_ev = (0 < hessian_ev).sum()
                    minima_line.extend([num_positive_ev, num_negative_ev])
                minima_line.extend(np.asarray(minima_parameters))
                minima_writer.writerow(minima_line)
        minima_file.close()

    # write corner points to file
    if options.cornerpoints_file is not None:
        header = trajectory_header[:]
        cornerpoints_writer, cornerpoints_file = setup_csv_file(
            options.cornerpoints_file, header)
        for id in range(len(cornerpoints)):
            for row in range(len(cornerpoints[id])):
                print(cornerpoints[id][row][0])
                print(cornerpoints[id][row][1])
                print(np.asarray(cornerpoints[id][row][2]))
                cornerpoints_line = [id, cornerpoints[id][row][0], cornerpoints[id][row][1]]
                cornerpoints_line.extend(np.asarray(cornerpoints[id][row][2]))
                print(cornerpoints_line)
                cornerpoints_writer.writerow(cornerpoints_line)
        cornerpoints_file.close()

    rt.set_overall_time(time.process_time() - time_zero)

def internal_main():
    global options

    # setup logging
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    unparsed = parse_parameters()

    options.react_to_common_options()
    options.react_to_sampler_options()

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
