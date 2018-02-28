import logging
from math import pow

from DataDrivenSampler.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_extract_minimium_candidates(TrajectoryJob):
    ''' This implements a job that extracts minimum candidates from the trajectory.

    This is done by looking at the gradients along the trajectory.
    '''

    SMALLEST_TOLERANCE_POWER = -6       # tolerance for gradients being small
    LARGEST_TOLERANCE_POWER = -1        # tolerance for gradients being small
    MINIMUM_EXTRACT_CANDIDATES = 3      # minimum number of candidates to extract


    def __init__(self, data_id, parameters, continue_flag = True):
        """ Initializes a extracting minimum candidates job.

        :param data_id: id associated with data object
        :param parameters: parameter for analysis
        :param continue_flag: flag allowing to override spawning of subsequent job
        """
        super(TrajectoryJob_extract_minimium_candidates, self).__init__(data_id)
        self.job_type = "extract_minimium_candidates"
        self.parameters = parameters
        self.continue_flag = continue_flag

    def run(self, _data):
        """ This implements extracting minimum candidates from the trajectory
        stored in a data object by looking at sections where the gradients is
        smaller than TOLERANCE and taking the smallest value in this section.

        :param _data: data object to use
        :param _object: FLAGS object that contains sampling parameters
        :return: updated data object
        """
        for i in range(self.SMALLEST_TOLERANCE_POWER, self.LARGEST_TOLERANCE_POWER+1):
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
                for j in range(small_gradient_start + 1, i):
                    if _data.gradients[j] < smallest_val:
                        smallest_val_index = j
                        smallest_val = _data.gradients[j]
                return smallest_val_index

            # gather new ones
            small_gradient_start = -1
            for i in range(1,len(_data.gradients)):
                if _data.gradients[i-1] > tolerance \
                    and _data.gradients[i] <= tolerance:
                    small_gradient_start = i
                elif _data.gradients[i-1] <= tolerance \
                    and _data.gradients[i] > tolerance:
                    if small_gradient_start != -1:
                        smallest_val_index = find_smallest_gradient_index(
                            small_gradient_start, i)
                        _data.minimum_candidates.append(smallest_val_index)
                        # reset gradient start
                        small_gradient_start = -1
            # check if last small gradient region extends till end of trajectory
            if (small_gradient_start != -1) and len(_data.minimum_candidates) != 0:
                smallest_val_index = find_smallest_gradient_index(small_gradient_start, -1)
                _data.minimum_candidates.append(smallest_val_index)

            if len(_data.minimum_candidates) > self.MINIMUM_EXTRACT_CANDIDATES:
                print("Picked "+str(len(_data.minimum_candidates))+" at threshold "+str(tolerance))
                break
            #else:
            #    print("Threshold " + str(tolerance)+" is too small still.")

        logging.info("Found minima candidates: "+str(_data.minimum_candidates))

        # after extract there is no other job (on that trajectory)
        return _data, False
