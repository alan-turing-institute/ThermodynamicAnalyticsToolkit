class Accumulator(object):
    """ This class defines the accumulator interface.
    """

    output_width = 8
    output_precision = 8

    def __init__(self):
        self._next_eval_step = None # next step when to write buffer
        self._buffer = []           # contains lines going to file or dataframe
        self._last_rejected = 0     # stores the last rejected from AccumulatedValues

    def accumulate_each_step(self, current_step):
        """ Accumulate values each step internally.

        :param current_step: current step with values
        """
        raise AttributeError("Not implemented, you need to derive properly from this class.")

    def accumulate_nth_step(self, current_step):
        """ Accumulate values each nth step, possibly writing to file.

        :param current_step: current step with values
        """
        raise AttributeError("Not implemented, you need to derive properly from this class.")

    def inform_next_eval_step(self, next_eval_step, rejected):
        self._next_eval_step = next_eval_step
        self._last_rejected = rejected