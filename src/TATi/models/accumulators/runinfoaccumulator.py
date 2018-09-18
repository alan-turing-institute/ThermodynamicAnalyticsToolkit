from TATi.models.accumulators.accumulator import Accumulator

import numpy as np
import pandas as pd
from math import sqrt

class RuninfoAccumulator(Accumulator):
    """ Here, we accumulate the run_info dataframe

    """

    def __init__(self, return_run_info, method, config_map, writer,
                 header, max_steps, every_nth, number_walkers):
        super(RuninfoAccumulator, self).__init__(method, max_steps, every_nth)
        self.run_info = None
        self._return_run_info = return_run_info
        self._config_map = config_map
        self._run_writer = writer
        self._number_walkers = number_walkers
        if self._return_run_info:
            self.run_info = []
            no_params = len(header)
            for walker_index in range(self._number_walkers):
                self.run_info.append(pd.DataFrame(
                    np.zeros((self._total_eval_steps, no_params)),
                    columns=header))

    def _accumulate_nth_step_line(self, current_step, walker_index, values):
        run_line = [walker_index, values.global_step[walker_index], current_step] \
                   + ['{:1.3f}'.format(values.accuracy[walker_index])] \
                   + ['{:{width}.{precision}e}'.format(values.loss[walker_index], width=self.output_width,
                                                       precision=self.output_precision)] \
                   + ['{:{width}.{precision}e}'.format(values.time_elapsed_per_nth_step, width=self.output_width,
                                                       precision=self.output_precision)]

        if self._method == "StochasticGradientLangevinDynamics" or self._method == "GradientDescent":
            run_line += ['{:{width}.{precision}e}'.format(x, width=self.output_width,
                                                          precision=self.output_precision)
                         for x in [sqrt(values.gradients[walker_index]), abs(0.5 * values.virials[walker_index])]]
            if self._method == "StochasticGradientLangevinDynamics":
                run_line += ['{:{width}.{precision}e}'.format(sqrt(values.noise[walker_index]), width=self.output_width,
                                                              precision=self.output_precision)]
        elif "HamiltonianMonteCarlo" in self._method:
            if (values.rejected[walker_index] + values.accepted[walker_index]) > 0:
                rejection_rate = values.rejected[walker_index] / (
                        values.rejected[walker_index] + values.accepted[walker_index])
            else:
                rejection_rate = 0
            run_line += ['{:{width}.{precision}e}'.format(values.total_energy[walker_index],
                                                          width=self.output_width,
                                                          precision=self.output_precision)] \
                        + ['{:{width}.{precision}e}'.format(values.last_old_total_energy[walker_index],
                                                            width=self.output_width,
                                                            precision=self.output_precision)] \
                        + ['{:{width}.{precision}e}'.format(x, width=self.output_width,
                                                            precision=self.output_precision)
                           for x in [values.kinetic_energy[walker_index], sqrt(values.momenta[walker_index]),
                                     sqrt(values.gradients[walker_index]), abs(0.5 * values.virials[walker_index])]] \
                        + ['{:{width}.{precision}e}'.format(rejection_rate, width=self.output_width,
                                                            precision=self.output_precision)]
        else:
            run_line += ['{:{width}.{precision}e}'.format(values.loss[walker_index] + values.kinetic_energy[walker_index],
                                                          width=self.output_width,
                                                          precision=self.output_precision)] \
                        + ['{:{width}.{precision}e}'.format(x, width=self.output_width,
                                                            precision=self.output_precision)
                           for x in [values.kinetic_energy[walker_index], sqrt(values.momenta[walker_index]),
                                     sqrt(values.gradients[walker_index]), abs(0.5 * values.virials[walker_index]),
                                     sqrt(values.noise[walker_index])]]
        return run_line

    def accumulate_nth_step(self, current_step, walker_index, values):
        if super(RuninfoAccumulator, self).accumulate_nth_step(current_step, walker_index):
            if self._config_map["do_write_run_file"] or self._return_run_info:
                run_line = []
                if self._method in ["GradientDescent",
                                    "StochasticGradientLangevinDynamics",
                                    "GeometricLangevinAlgorithm_1stOrder",
                                    "GeometricLangevinAlgorithm_2ndOrder",
                                    "HamiltonianMonteCarlo_1stOrder",
                                    "HamiltonianMonteCarlo_2ndOrder",
                                    "BAOAB",
                                    "CovarianceControlledAdaptiveLangevinThermostat"]:
                    run_line = self._accumulate_nth_step_line(current_step, walker_index, values)
                if self._config_map["do_write_run_file"] and self._run_writer is not None:
                    self._run_writer.writerow(run_line)
                if self._return_run_info:
                    self.run_info[walker_index].loc[self.written_row] = run_line
                self.written_row +=1