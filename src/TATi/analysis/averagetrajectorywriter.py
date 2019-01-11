import numpy as np

from TATi.common import setup_csv_file
from TATi.analysis.parsedtrajectory import ParsedTrajectory

class AverageTrajectoryWriter(object):
    """ This class writes the averaged parameters to a CSV file

    """

    output_width = 8
    output_precision = 8

    def __init__(self, trajectory):
        if isinstance(trajectory, ParsedTrajectory):
            self._trajectory = trajectory.get_trajectory()
        else:
            self._trajectory = trajectory
        self.number_dof = len(self._trajectory[0,:])
        self.average_params = [np.average(self._trajectory[0:, i]) for i in range(self.number_dof)]
        self.variance_params = [np.var(self._trajectory[0:, i]) for i in range(self.number_dof)]
        #print("First ten parameters are converged to the following values:")
        #print(str(self.average_params[0:10]))
        #print(str(self.variance_params[0:10]))

    def write(self, filename):
        """ Write average and variance of each parameter to file.

        :param filename: filename of file
        """
        csv_writer, csv_file = setup_csv_file(filename, ['step', 'average_parameter', 'variance_parameter'])
        for step, avg,var in zip(range(self.number_dof), self.average_params, self.variance_params):
            csv_writer.writerow(
                [step, '{:{width}.{precision}e}'.format(avg, width=self.output_width,
                                                        precision=self.output_precision)]+
                ['{:{width}.{precision}e}'.format(var, width=self.output_width,
                                                  precision=self.output_precision)]
            )
        csv_file.close()
