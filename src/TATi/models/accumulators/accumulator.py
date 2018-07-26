class Accumulator(object):
    """ This class defines the accumulator interface.
    """

    output_width = 8
    output_precision = 8

    def __init__(self):
        pass

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