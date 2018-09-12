import logging

class Accumulator(object):
    """ This class defines the accumulator interface.
    """

    output_width = 8
    output_precision = 8

    def __init__(self, every_nth):
        self._next_eval_step = []   # next step when to write buffer
        self._last_rejected = 0     # stores the last rejected from AccumulatedValues
        self._every_nth = every_nth # stores that only each nth output step is actually written
        self._internal_nth = 0      # internal counting for dropping other but nth step
        self.written_row = 0        # current row to append in accumulated lines

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
        else:
            # here we count to only write every_nth step
            self._internal_nth += 1
            if self._internal_nth == self._every_nth:
                self._internal_nth = 0

            return self._internal_nth == 0

    def inform_next_eval_step(self, next_eval_step, rejected):
        self._next_eval_step[:] = next_eval_step
        self._last_rejected = rejected
