from TATi.models.accumulators.accumulator import Accumulator

import numpy as np
import pandas as pd
from math import sqrt

class RuninfoAccumulator(Accumulator):
    """ Here, we accumulate the run_info dataframe

    """

    def __init__(self, return_run_info, sampler, config_map, writer,
                 header, steps, number_walkers):
        super(RuninfoAccumulator, self).__init__()
        self.run_info = None
        self._return_run_info = return_run_info
        self._sampler = sampler
        self._config_map = config_map
        self._run_writer = writer
        self._number_walkers = number_walkers
        if self._return_run_info:
            self.run_info = []
            no_params = len(header)
            for walker_index in range(self._number_walkers):
                self.run_info.append(pd.DataFrame(
                    np.zeros((steps, no_params)),
                    columns=header))

    def _accumulate_nth_step_line(self, current_step, walker_index, written_row, values):
        run_line = [walker_index, values.global_step[walker_index], current_step] \
                   + ['{:1.3f}'.format(values.accuracy[walker_index])] \
                   + ['{:{width}.{precision}e}'.format(values.loss[walker_index], width=self.output_width,
                                                       precision=self.output_precision)] \
                   + ['{:{width}.{precision}e}'.format(values.time_elapsed_per_nth_step, width=self.output_width,
                                                       precision=self.output_precision)]

        if self._sampler == "StochasticGradientLangevinDynamics":
            run_line += ['{:{width}.{precision}e}'.format(x, width=self.output_width,
                                                          precision=self.output_precision)
                         for x in [sqrt(values.gradients[walker_index]), abs(0.5 * values.virials[walker_index]),
                                   sqrt(values.noise[walker_index])]]
        elif self._sampler == "HamiltonianMonteCarlo":
            if (values.rejected[walker_index] + values.accepted[walker_index]) > 0:
                rejection_rate = values.rejected[walker_index] / (
                        values.rejected[walker_index] + values.accepted[walker_index])
            else:
                rejection_rate = 0
            run_line += ['{:{width}.{precision}e}'.format(values.loss[walker_index] + values.kinetic_energy[walker_index],
                                                          width=self.output_width,
                                                          precision=self.output_precision)] \
                        + ['{:{width}.{precision}e}'.format(values.old_total_energy[walker_index],
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


    def accumulate_nth_step(self, current_step, walker_index, written_row, values):
        run_line = []
        if self._sampler in ["StochasticGradientLangevinDynamics",
                             "GeometricLangevinAlgorithm_1stOrder",
                             "GeometricLangevinAlgorithm_2ndOrder",
                             "HamiltonianMonteCarlo",
                             "BAOAB",
                             "CovarianceControlledAdaptiveLangevinThermostat"]:
            run_line = self._accumulate_nth_step_line(current_step, walker_index, written_row, values)
        if self._config_map["do_write_run_file"]:
            self._run_writer.writerow(run_line)
        if self._return_run_info:
            self.run_info[walker_index].loc[written_row] = run_line
