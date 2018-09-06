class AccumulatedValues(object):
    """ This is a simple structure holding a few values needed for accumulation.

    """

    def __init__(self):
        self.loss = None
        self.accuracy = None

        self.weights = None
        self.biases = None

        self.time_elapsed_per_nth_step = None

        self.gradients = None
        self.virials = None
        self.noise = None
        self.kinetic_energy = None
        self.momenta = None

        # only for HMC
        self.total_energy = []
        self.old_total_energy = None
        self.last_old_total_energy = None   # accept overwrites old_total_energy, hence keep last
        self.old_kinetic_energy = None    # temporary for delaying kinetic energy by one ste
        self.accepted = 0
        self.rejected = 0

    def evaluate(self, sess, sampler, static_vars):
        if sampler in ["StochasticGradientLangevinDynamics",
                                  "GeometricLangevinAlgorithm_1stOrder",
                                  "GeometricLangevinAlgorithm_2ndOrder",
                                  "HamiltonianMonteCarlo_1stOrder",
                                  "BAOAB",
                                  "CovarianceControlledAdaptiveLangevinThermostat"]:
            if sampler == "StochasticGradientLangevinDynamics":
                self.gradients, self.virials, self.noise = \
                    sess.run([
                        static_vars["gradients"],
                        static_vars["virials"],
                        static_vars["noise"]])
            elif sampler == "HamiltonianMonteCarlo_1stOrder":
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
                    self.kinetic_energy, self.momenta, self.gradients, self.virials = \
                    sess.run([
                        static_vars["accepted"],
                        static_vars["rejected"],
                        static_vars["old_total_energy"],
                        static_vars["kinetic_energy"],
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
                self.kinetic_energy, self.momenta, self.gradients, self.virials, self.noise = \
                    sess.run([
                        static_vars["kinetic_energy"],
                        static_vars["momenta"],
                        static_vars["gradients"],
                        static_vars["virials"],
                        static_vars["noise"]])