import logging
from math import pow

from TATi.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_extract_minimium_candidates(TrajectoryJob):
    ''' This implements a job that extracts minimum candidates from the trajectory.

    We look through the thresholds 10^[largest_power, smallest_power] in order
    find sufficiently many but not too many. Therefore, we look for small
    gradients along the trajectory starting from the smallest power and stop
    as soon as we find sufficiently many.

    Candidates are stored in the trajectorydata's minimum_candidates variable.
    '''

    SMALLEST_TOLERANCE_POWER = -6       # tolerance for gradients being small
    LARGEST_TOLERANCE_POWER = -1        # tolerance for gradients being small
    MINIMUM_EXTRACT_CANDIDATES = 3      # minimum number of candidates to extract


    def __init__(self, data_id, parameters, smallest_power=None, largest_power=None, continue_flag = True):
        """ Initializes a extracting minimum candidates job.

        :param data_id: id associated with data object
        :param parameters: parameter for analysis
        :param smallest_power: smallest power for tolerance search, None - use default
        :param largest_power: largest power for tolerance search, None - use default
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_extract_minimium_candidates, self).__init__(data_id)
        self.job_type = "extract_minimium_candidates"
        self.parameters = parameters
        if smallest_power is None:
            self.smallest_power = self.SMALLEST_TOLERANCE_POWER
        else:
            self.smallest_power = smallest_power
        if largest_power is None:
            self.largest_power = self.LARGEST_TOLERANCE_POWER
        else:
            self.largest_power = largest_power
        self.continue_flag = continue_flag

    def run(self, _data):
        """ This implements extracting minimum candidates from the trajectory
        stored in a data object by looking at sections where the gradients is
        smaller than TOLERANCE and taking the smallest value in this section.

        :param _data: data object to use
        :param _object: FLAGS object that contains sampling parameters
        :return: updated data object
        """
        for i in range(self.smallest_power, self.largest_power+1):
            tolerance = pow(10,i)
            # delete already present ones
            _data.minimum_candidates[:] = []

            def find_smallest_gradient_index(small_gradient_start, small_gradient_end):
                """ Helper function to find the smallest value in the given interval.

                :param small_gradient_start: first value (included) of interval
                :param small_gradient_end: last value (excluded) of interval
                :return: index to the smallest value in the region
                """
                smallest_val_index = small_gradient_start
                smallest_val = _data.gradients[smallest_val_index]
                for j in range(small_gradient_start + 1, small_gradient_end):
                    if _data.gradients[j] < smallest_val:
                        smallest_val_index = j
                        smallest_val = _data.gradients[j]
                return smallest_val_index

            # gather new ones
            if _data.gradients[0] <= tolerance:
                small_gradient_start = 0
            else:
                small_gradient_start = -1
            for i in range(1,len(_data.gradients)):
                if _data.gradients[i-1] > tolerance \
                    and _data.gradients[i] <= tolerance:
                    logging.debug("Found start of region below tolerance "+str(tolerance) \
                                  +" at "+str(i)+" with "+str(_data.gradients[i]))
                    small_gradient_start = i
                elif _data.gradients[i-1] <= tolerance \
                    and _data.gradients[i] > tolerance:
                    logging.debug("Found end of region below tolerance "+str(tolerance) \
                                  +" at "+str(i)+" with "+str(_data.gradients[i]))
                    if small_gradient_start != -1:
                        smallest_val_index = find_smallest_gradient_index(
                            small_gradient_start, i)
                        _data.minimum_candidates.append(smallest_val_index)
                        logging.debug("Picked " + str(smallest_val_index)+ " with " \
                                      +str(_data.gradients[smallest_val_index]) \
                                      +" as candidate from region")
                        # reset gradient start
                        small_gradient_start = -1
            # check if last small gradient region extends till end of trajectory
            if (small_gradient_start != -1) and len(_data.gradients) != 0:
                smallest_val_index = find_smallest_gradient_index(
                    small_gradient_start, len(_data.gradients))
                _data.minimum_candidates.append(smallest_val_index)
                logging.debug("Picked " + str(smallest_val_index) + " with " \
                              + str(_data.gradients[smallest_val_index]) \
                              + " as candidate from last region")

            if len(_data.minimum_candidates) > self.MINIMUM_EXTRACT_CANDIDATES:
                logging.info("Picked "+str(len(_data.minimum_candidates))+" at threshold "+str(tolerance))
                break

        logging.info("Found minima candidates: "+str(_data.minimum_candidates))

        # after extract there is no other job (on that trajectory)
        return _data, len(_data.minimum_candidates) > 0
