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
from math import pow

from TATi.exploration.trajectoryjob import TrajectoryJob

class TrajectoryJob_extract_minimium_candidates(TrajectoryJob):
    """This implements a job that extracts minimum candidates from the trajectory.
    
    This is done by looking at the gradients along the trajectory.

    Args:

    Returns:

    """

    SMALLEST_TOLERANCE_POWER = -6       # tolerance for gradients being small
    LARGEST_TOLERANCE_POWER = -1        # tolerance for gradients being small
    MINIMUM_EXTRACT_CANDIDATES = 3      # minimum number of candidates to extract


    def __init__(self, data_id, parameters, continue_flag = True):
        """Initializes a extracting minimum candidates job.

        Args:
          data_id: id associated with data object
          parameters: parameter for analysis
          continue_flag: flag allowing to override spawning of subsequent job (Default value = True)

        Returns:

        """
        super(TrajectoryJob_extract_minimium_candidates, self).__init__(data_id)
        self.job_type = "extract_minimium_candidates"
        self.parameters = parameters
        self.continue_flag = continue_flag

    def run(self, _data):
        """This implements extracting minimum candidates from the trajectory
        stored in a data object by looking at sections where the gradients is
        smaller than TOLERANCE and taking the smallest value in this section.

        Args:
          _data: data object to use

        Returns:
          updated data object

        """
        for i in range(self.SMALLEST_TOLERANCE_POWER, self.LARGEST_TOLERANCE_POWER+1):
            tolerance = pow(10,i)
            # delete already present ones
            _data.minimum_candidates[:] = []

            def find_smallest_gradient_index(small_gradient_start, small_gradient_end):
                """Helper function to find the smallest value in the given interval.

                Args:
                  small_gradient_start: first value (included) of interval
                  small_gradient_end: last value (excluded) of interval

                Returns:
                  index to the smallest value in the region

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
