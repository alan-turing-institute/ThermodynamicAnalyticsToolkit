from TATi.models.accumulators.accumulator import Accumulator

from math import exp
import numpy as np
import pandas as pd

class AveragesAccumulator(Accumulator):
    """ This class takes care of accumulating averages properly.

    """

    def __init__(self, return_averages, method, config_map, writer,
                 header, max_steps, every_nth, number_walkers,
                 inverse_temperature=1., burn_in_steps=0):
        super(AveragesAccumulator, self).__init__(method, max_steps, every_nth)
        self.accumulated_kinetic_energy = [0.]*number_walkers
        self.accumulated_loss_nominator = [0.]*number_walkers
        self.accumulated_loss_denominator = [0.]*number_walkers
        self.accumulated_virials = [0.]*number_walkers
        self.accumulated_inertia = [0.]*number_walkers
        self.last_inertia = [0.]*number_walkers
        self.averages = None
        self._return_averages = return_averages
        self._config_map = config_map
        self._averages_writer = writer

        self._number_walkers = number_walkers
        self._burn_in_steps = burn_in_steps
        self._inverse_temperature = inverse_temperature

        self.accumulated_steps = 0

        if self._return_averages:
            self.averages = []
            no_params = len(header)
            for walker_index in range(self._number_walkers):
                self.averages.append(pd.DataFrame(
                    np.zeros((self._total_eval_steps, no_params)),
                    columns=header))

    def accumulate_each_step(self, current_step, walker_index, values):
        if current_step >= self._burn_in_steps:
            self.accumulated_steps += 1
            self.accumulated_loss_nominator[walker_index] += values.loss[walker_index] * exp(
                - self._inverse_temperature * values.loss[walker_index])
            self.accumulated_loss_denominator[walker_index] += exp(
                - self._inverse_temperature * values.loss[walker_index])
            self.accumulated_virials[walker_index] += values.virials[walker_index]
            if self._method != "StochasticGradientLangevinDynamics" and self._method != "GradientDescent":
                self.accumulated_kinetic_energy[walker_index] += values.kinetic_energy[walker_index]
                if self.accumulated_steps > 1:
                    inertia_secant = (values.inertia[walker_index] - self.last_inertia[walker_index])
                    self.accumulated_inertia[walker_index] += inertia_secant
                self.last_inertia[walker_index] = values.inertia[walker_index]

    def _accumulate_nth_step_line(self, current_step, walker_index, values):
        if self.accumulated_loss_denominator[walker_index] > 0:
            average_loss = self.accumulated_loss_nominator[walker_index] / self.accumulated_loss_denominator[
                walker_index]
        else:
            average_loss = 0.

        divisor = float(self.accumulated_steps)
        if divisor > 0.:
            average_kinetic_energy = self.accumulated_kinetic_energy[walker_index] / divisor
            average_virials = abs(0.5 * self.accumulated_virials[walker_index]) / divisor
            if (divisor-1.) > 0.:
                average_inertia = self.accumulated_inertia[walker_index] / (divisor-1.)
            else:
                average_inertia = 0.
        else:
            average_kinetic_energy = 0.
            average_virials = 0.
            average_inertia = 0.

        averages_line = [walker_index, values.global_step[walker_index], current_step] \
                        + ['{:{width}.{precision}e}'.format(values.loss[walker_index], width=self.output_width,
                                                            precision=self.output_precision)]
        if self._method != "GradientDescent":
            averages_line += ['{:{width}.{precision}e}'.format(average_loss, width=self.output_width,
                                                                precision=self.output_precision)]

        if self._method == "StochasticGradientLangevinDynamics" or self._method == "GradientDescent":
            averages_line += ['{:{width}.{precision}e}'.format(average_virials, width=self.output_width,
                                                               precision=self.output_precision)]
        else:
            averages_line += ['{:{width}.{precision}e}'.format(x, width=self.output_width,
                                                               precision=self.output_precision)
                              for x in [average_kinetic_energy, average_virials, average_inertia]]
        if "HamiltonianMonteCarlo" in self._method:
            if (values.rejected[walker_index] + values.accepted[walker_index]) > 0:
                average_rejection_rate = values.rejected[walker_index] / (
                        values.rejected[walker_index] + values.accepted[walker_index])
            else:
                average_rejection_rate = 0
            averages_line += ['{:{width}.{precision}e}'.format(average_rejection_rate, width=self.output_width,
                                                               precision=self.output_precision)]
        return averages_line

    def accumulate_nth_step(self, current_step, walker_index, values):
        if super(AveragesAccumulator, self).accumulate_nth_step(current_step, walker_index):
            if self._config_map["do_write_averages_file"] or self._return_averages:
                averages_line = self._accumulate_nth_step_line(current_step, walker_index, values)
                if self._config_map["do_write_averages_file"] and self._averages_writer is not None:
                    self._averages_writer.writerow(averages_line)
                if self._return_averages:
                    self.averages[walker_index].loc[self.written_row] = averages_line
                self.written_row +=1