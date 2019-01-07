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
import tempfile

from TATi.exploration.trajectoryjob import TrajectoryJob
from TATi.common import setup_csv_file

class TrajectoryProcess(TrajectoryJob):
    ''' This is the base class for process object that can be placed in the
    TrajectoryQueue for processing. In contrast to a trajectory job the
    process may be run independently, even on another host.

    This class needs to be derived and a proper run() method set up and the
    type of the job set.

    '''

    def __init__(self, data_id, network_model):
        """ Initializes the trajectory process.

        :param _data_id: id associated with data object
        :param network_model: neural network object for creating model files as starting points
        """
        super(TrajectoryProcess, self).__init__(data_id)
        self.network_model = network_model

    def create_starting_parameters(self, _data, number_weights, number_biases):
        # create model files from the parameters
        assert( len(_data.parameters) != 0 )
        parameters = _data.parameters[-1]
        logging.debug("Writing initial parameters " \
              +str(parameters[0:5])+" at step " \
              +str(_data.steps[-1])+" to temporary file.")
        f = tempfile.NamedTemporaryFile(mode="w", prefix="parameters-", suffix=".csv")
        filename = f.name
        f.close()
        header = ['step']\
           + [str("weight")+str(i) for i in range(0, number_weights)]\
           + [str("bias") + str(i) for i in range(0, number_biases)]

        parameters_writer, parameters_file = setup_csv_file(
            filename, header)
        write_row = [0, _data.steps[-1]]
        write_row.extend(parameters)
        parameters_writer.writerow(write_row)
        parameters_file.close()
        return filename, _data.steps[-1]

    @staticmethod
    def get_options_from_flags(FLAGS, keys):
        """

        :param FLAGS: set of parameters
        :param keys: set of keys from FLAGS to extract as command-line parameters
        :return: list of parameters for a process to start
        """
        options = []
        for key in keys:
            attribute = getattr(FLAGS, key)
            if attribute is not None:
                if isinstance(attribute, list):
                    # print only non-empty lists
                    if len(attribute) != 0:
                        options.extend(["--"+key, " ".join(attribute)])
                elif isinstance(attribute, bool):
                    # print bools numerically
                    options.extend(["--"+key, "1" if attribute else "0"])
                elif isinstance(attribute, str):
                    # print only non-empty strings
                    if len(attribute) != 0:
                        options.extend(["--"+key, attribute])
                else:
                    options.extend(["--" + key, str(attribute)])
        return options
