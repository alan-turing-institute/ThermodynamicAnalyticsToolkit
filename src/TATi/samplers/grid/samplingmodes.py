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

import inspect
import logging

from TATi.samplers.grid.naivegridsampler import NaiveGridSampler
from TATi.samplers.grid.subgridsampler import SubgridSampler
from TATi.samplers.grid.trajectoryresampler import TrajectoryReSampler
from TATi.samplers.grid.trajectorysubspaceresampler import TrajectorySubspaceReSampler


class SamplingModes(object):
    """ This class contains all possible sampling modes for
    TATiLossFunctionSampler as functions that all share the specific
    prefix "sample_" in their name.

    By this naming scheme, we can easily extract all available modes by class'
    inspection. This works especially when new modes are added -- simply by
    adding another function adhering to this naming convention.

    Hence, this is a factory for the samplers.

    """
    @staticmethod
    def list_modes():
        """ Returns a list of available mode names.

        :return: list of all available mode names
        """
        modes = SamplingModes._get_modes()
        return sorted([funcname.replace("_sample_", "") for funcname in modes.keys()])

    @staticmethod
    def create(mode, network_model, options):
        """ Factory's create function to build the sampler based on \a mode.

        The center point in "grid" sampling is taken from \a options.trajectory_file
        and the row specified by \a options.trajectory_stepnr. In case of
        "trajectory" sampling, no \a trajectory_stepnr is needed.

        :param mode: name of the mode (see samplingmodes.modes for dict whose keys give
            all available names)
        :param network_model: network model whose loss to sample
        :param options: options dict containing all option required by the samplers
        :return: created sampler object or None if illegal mode name
        """
        trajectory_file = options.parse_parameters_file
        if len(options.parse_steps) == 0:
            trajectory_stepnr = None
        else:
            trajectory_stepnr = options.parse_steps[0]
        funcname = "_sample_"+mode
        modes = SamplingModes._get_modes()
        if funcname in modes.keys():
            return modes[funcname](
                network_model, options, trajectory_file, trajectory_stepnr)
        else:
            return None

    @staticmethod
    def _get_modes():
        modes = { name: func
                  for (name, func) in inspect.getmembers(SamplingModes, predicate=inspect.isroutine)
                  if "_sample_" in name }
        return modes

    @staticmethod
    def _sample_naive_grid(network_model, options, trajectory_file, trajectory_stepnr):
        """ Creates a "naive grid" sampler

        :param network_model: network model whose loss to sample
        :param options: options dict containing all option required by the samplers
        :param trajectory_file: trajectory file name
        :param trajectory_stepnr: step number to use (designating row) inside \a trajectory_file
        :return: created sampler object
        """
        sampler = NaiveGridSampler(network_model=network_model,
                                   exclude_parameters=options.exclude_parameters,
                                   samples_weights=options.samples_weights,
                                   samples_biases=options.samples_biases)
        sampler.setup_start(trajectory_file=trajectory_file,
                            trajectory_stepnr=trajectory_stepnr,
                            interval_weights=options.interval_weights,
                            interval_biases=options.interval_biases)
        return sampler

    @staticmethod
    def _sample_naive_subgrid(network_model, options, trajectory_file, trajectory_stepnr):
        """ Creates a "naive subgrid" sampler that takes a directions file into
        account wherein row vectors span the subspace to sample.

        :param network_model: network model whose loss to sample
        :param options: options dict containing all option required by the samplers
        :param trajectory_file: trajectory file name
        :param trajectory_stepnr: step number to use (designating row) inside \a trajectory_file
1        :return: created sampler object
        """
        sampler = SubgridSampler.from_file(
                network_model=network_model,
                exclude_parameters=options.exclude_parameters,
                samples_weights=options.samples_weights,
                directions_file=options.directions_file)
        sampler.setup_start(trajectory_file=trajectory_file,
                            trajectory_stepnr=trajectory_stepnr,
                            interval_weights=options.interval_weights,
                            interval_offsets=options.interval_offsets)
        return sampler

    @staticmethod
    def _common_trajectory(network_model, options, trajectory_file, directions_file):
        """ This helper function instantiates a "trajectory" sampler

        :param network_model: network model whose loss to sample
        :param options: options dict containing all option required by the samplers
        :param trajectory_file: trajectory file name
        :param directions_file: None - all directions, else - only subspace spanned by row in this file
        :return: created sampler object
        """
        # we cover both trajectory and trajectory_subgrid in this statement

    @staticmethod
    def _sample_trajectory(network_model, options, trajectory_file, _):
        """ Creates a "trajectory" sampler.

        :param network_model: network model whose loss to sample
        :param options: options dict containing all option required by the samplers
        :param trajectory_file: trajectory file name
        :return: created sampler object
        """
        if len(options.parse_steps) != 0:
            logging.warning("Option parse_steps is not used when sampling in mode trajectory.")
        sampler = TrajectoryReSampler.from_trajectory_file(
                network_model=network_model,
                exclude_parameters=options.exclude_parameters,
                trajectory_file=trajectory_file)
        sampler.setup_start()
        return sampler

    @staticmethod
    def _sample_trajectory_subgrid(network_model, options, trajectory_file, _):
        """ Creates a "trajectory" sampler that stores only the coordinates of the given
        subgrid (rows in \a directions_file)

        :param network_model: network model whose loss to sample
        :param options: options dict containing all option required by the samplers
        :param trajectory_file: trajectory file name
        :return: created sampler object
        """
        if len(options.parse_steps) != 0:
            logging.warning("Option parse_steps is not used when sampling in mode trajectory.")
        if len(options.directions_file) is None:
            raise ValueError("Mode 'trajectory_subgrid' requires set directions_file.")
        sampler = TrajectorySubspaceReSampler.from_files(
                network_model=network_model,
                exclude_parameters=options.exclude_parameters,
                trajectory_file=trajectory_file,
                directions_file=options.directions_file)
        sampler.setup_start()
        return sampler
