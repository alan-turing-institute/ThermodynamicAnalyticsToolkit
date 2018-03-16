import logging
import os

from DataDrivenSampler.exploration.trajectoryjob import TrajectoryJob


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

    def _set_parameters(self, parameters):
        # set parameters to ones from old leg (if exists)
        sess = self.network_model.sess
        weights_dof = self.network_model.weights.get_total_dof()
        self.network_model.weights.assign(sess, self.parameters[0:weights_dof])
        self.network_model.biases.assign(sess, parameters[weights_dof:])

    def create_starting_model(self, _data, model_filename):
        foldername = os.path.dirname(model_filename)
        # save starting parameters set to a model
        if not os.path.isdir(foldername):
            os.mkdir(foldername)
            logging.debug("Creating folder "+foldername)
            # create model files from the parameters
            assert( len(_data.parameters) != 0 )
            print("Create initial model from parameters "+str(_data.parameters[-1][0:5]))
            parameters = _data.parameters[-1]
            self._set_parameters(parameters)
            save_path = self.network_model.saver.save(
                self.network_model.sess, model_filename)

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
