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

class TrajectoryJob(object):
    """This is the base class for job object that can be placed in the
    TrajectoryQueue for processing.
    
    This class needs to be derived and a proper run() method set up and the
    type of the job set.

    Args:

    Returns:

    """

    def __init__(self, data_id):
        """Initializes the trajectory job.

        Args:
          data_id: id associated with data object

        Returns:

        """
        self.data_id = data_id
        self.job_id = -1
        self.job_type = "generic"

    def set_job_id(self, job_id):
        """Sets the job id for this job. Can only be done once.

        Args:
          job_id: new id of the job

        Returns:

        """
        # assert its unset so far
        assert( self.job_id == -1 )
        self.job_id = job_id

    def get_data_id(self):
        """

        Args:

        Returns:
            unique id of data object

        """
        return self.data_id

    def get_job_id(self):
        """

        Args:

        Returns:
            unique id of object

        """
        return self.job_id

    def run(self, _data, _object=None):
        """This function needs to overriden.

        Args:
          _data: data object to use
          _object: additional run object (Default value = None)

        Returns:
          updated data object

        """
        assert( 0 )