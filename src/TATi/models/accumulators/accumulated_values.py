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

class AccumulatedValues(object):
    """This is a simple structure holding a few values needed for accumulation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = None
        self.accuracy = None
        self.global_step = None

        self.weights = None
        self.biases = None

        self.time_elapsed_per_nth_step = 0.

        self.gradients = None
        self.virials = None
        self.noise = None
        self.kinetic_energy = None
        self.momenta = None
        self.inertia = None

        # only for Optimizers
        self.learning_rate_current = None

        # only for HMC
        self.total_energy = []
        self.old_total_energy = None
        self.last_old_total_energy = None   # accept overwrites old_total_energy, hence keep last
        self.old_kinetic_energy = None    # temporary for delaying kinetic energy by one ste
        self.accepted = 0
        self.rejected = 0

    def evaluate(self, sess, method, static_vars):
        if method in ["BarzilaiBorweinGradientDescent",
                      "GradientDescent",
                      "StochasticGradientLangevinDynamics",
                      "GeometricLangevinAlgorithm_1stOrder",
                      "GeometricLangevinAlgorithm_2ndOrder",
                      "HamiltonianMonteCarlo_1stOrder",
                      "HamiltonianMonteCarlo_2ndOrder",
                      "BAOAB",
                      "CovarianceControlledAdaptiveLangevinThermostat"]:
            if method == "StochasticGradientLangevinDynamics" or "GradientDescent" in method:
                self.gradients, self.virials, self.noise, temp, temp_dim = \
                    sess.run([
                        static_vars["gradients"],
                        static_vars["virials"],
                        static_vars["noise"],
                        static_vars["learning_rate_current"],
                        static_vars["learning_rate_current_dim"]])
                if "GradientDescent" in method:
                    self.learning_rate_current = temp
                    for i in range(len(temp)):
                        if temp_dim[i] != 0:
                            self.learning_rate_current[i] = self.learning_rate_current[i]/temp_dim[i]
            elif "HamiltonianMonteCarlo" in method:
                # when HMC accepts, it overwrites `old_total_energy` with the updated value
                # hence, we cannot see the old value in output any more. Therefore, we
                # always keep the last value as backup.
                if self.old_total_energy is not None:
                    self.last_old_total_energy[:] = self.old_total_energy

                # kinetic energy is ahead by one step of loss, therefore we need to sum
                # the loss with the old kinetic energy to get the energy of the proposed state
                if self.old_kinetic_energy is not None:
                    self.old_kinetic_energy[:] = self.kinetic_energy
                else:
                    # for very first evaluation step we still have zero kinetic energy and
                    # need to create the array
                    self.old_kinetic_energy = [0.] * len(self.kinetic_energy)

                self.accepted, self.rejected, self.old_total_energy, \
                    self.kinetic_energy, self.inertia, self.momenta, self.gradients, self.virials = \
                    sess.run([
                        static_vars["accepted"],
                        static_vars["rejected"],
                        static_vars["old_total_energy"],
                        static_vars["kinetic_energy"],
                        static_vars["inertia"],
                        static_vars["momenta"],
                        static_vars["gradients"],
                        static_vars["virials"]])

                # in the first step we simply copy it to properly initialize the last value
                if self.last_old_total_energy is None:
                    self.last_old_total_energy = self.old_total_energy.copy()

                # total energy is the current loss with the old kinetic energy. We sum
                # it up here in order to be able to output the energies of the initial and
                # the proposed state properly
                self.total_energy[:] = self.loss
                for walker_index in range(len(self.loss)):
                    self.total_energy[walker_index] += self.old_kinetic_energy[walker_index]
            else:
                self.kinetic_energy, self.inertia, self.momenta, self.gradients, self.virials, self.noise = \
                    sess.run([
                        static_vars["kinetic_energy"],
                        static_vars["inertia"],
                        static_vars["momenta"],
                        static_vars["gradients"],
                        static_vars["virials"],
                        static_vars["noise"]])