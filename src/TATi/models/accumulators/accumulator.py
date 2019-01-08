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

class Accumulator(object):
    """ This class defines the accumulator interface.

    Example:
        >> acc = AveragesAccumulator(...)
        >> acc.reset(true, [""])

        During each step of the iteration call

        >> acc.accumulate_each_step(current_step)

        During each nth step (to print to file or screen)

        >> acc.accumulate_nth_step(current_step)

        For HMC sampling you need to inform about the next acceptance criterion
        evaluation like

        >> acc.inform_next_eval_step(...)
    """

    output_width = 8
    output_precision = 8

    def __init__(self, method, max_steps, every_nth, number_walkers):
        self._method = method           # stores the sampling/optimization method
        self._max_steps = max_steps     # stores which total number of steps are evaluated
        self._every_nth = every_nth     # stores that only each nth output step is actually written
        self._number_walkers = number_walkers

    def reset(self):
        self._next_eval_step = []       # next step when to write buffer
        self._last_rejected = 0         # stores the last rejected from AccumulatedValues
        self._internal_nth = [-1]*self._number_walkers         # internal counting for dropping other but nth step
        self.written_row = 0            # current row to append in accumulated lines
        self._total_eval_steps = (self._max_steps % self._every_nth) + 1

    def init_writer(self, writer):
        self._writer = writer

    def accumulate_each_step(self, current_step):
        """ Accumulate values each step internally.

        :param current_step: current step with values
        """
        raise AttributeError("Not implemented, you need to derive properly from this class.")

    def accumulate_nth_step(self, current_step, walker_index):
        """ Accumulate values each nth step, possibly writing to file.

        Here, we centrally control for all derived accumulators whether something
        should be written this particular `current_step` or not.

        :param current_step: current step with values
        :param walker_index: index of the walker in question
        :return: True - write something, False - do not
        """
        # here, we check whether we are past an acceptance criterion evaluation and
        # hence need to write something
        if ((len(self._next_eval_step) > 0) and (current_step != self._next_eval_step[walker_index])):
            # HMC (_next_eval_step is non-empty) outside evaluation step
            return False
        if (self._method == "GradientDescent") and (current_step == (self._max_steps-1)):
            return True

        # here we count to only write every_nth step
        self._internal_nth[walker_index] += 1
        if self._internal_nth[walker_index] == self._every_nth:
            self._internal_nth[walker_index] = 0

        return self._internal_nth[walker_index] == 0

    def inform_next_eval_step(self, next_eval_step, rejected):
        self._next_eval_step[:] = next_eval_step
        self._last_rejected = rejected
