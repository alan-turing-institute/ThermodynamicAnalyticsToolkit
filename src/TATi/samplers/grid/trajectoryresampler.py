import pandas as pd

from TATi.samplers.grid.sampler import Sampler

class TrajectoryReSampler(Sampler):
    """ This class implements a sampler that simply re-evaluates all points
    of a given trajectory.

    This can be useful if the points are to be evaluated on a different loss
    function or dataset (e.g., test data set in place of training data set).

    """
    def __init__(self, network_model, exclude_parameters, df_trajectory):
        super(TrajectoryReSampler, self).__init__(network_model=network_model,
                                                  exclude_parameters=exclude_parameters)
        self.df_trajectory = df_trajectory

    @classmethod
    def from_trajectory_file(cls, network_model, exclude_parameters, trajectory_file):
        df_trajectory = pd.read_csv(trajectory_file, sep=',', header=0)
        return cls(network_model=network_model,
                   exclude_parameters=exclude_parameters,
                   df_trajectory=df_trajectory)

    def get_max_steps(self):
        return len(self.df_trajectory.index)

    def _prepare_header(self):
        header = super(TrajectoryReSampler, self)._prepare_header()
        return self._add_all_degree_header(header)

    def setup_start(self):
        pass

    def goto_start(self):
        super(TrajectoryReSampler, self).goto_start()
        self.rownr = self.df_trajectory.index[self.current_step]

    def set_step(self):
        weights_eval, biases_eval = self.network_model.assign_weights_and_biases_from_dataframe(
            df_parameters=self.df_trajectory,
            rownr=self.rownr,
            do_check=True
        )
        return self._combine_into_coords(weights_eval, biases_eval)

    def goto_next_step(self):
        super(TrajectoryReSampler, self).goto_next_step()
        if self.current_step < len(self.df_trajectory.index):
            self.rownr = self.df_trajectory.index[self.current_step]


