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
        self.old_total_energy = None
        self.kinetic_energy = None
        self.momenta = None

        self.accepted = None
        self.rejected = None

    def evaluate(self, sess, sampler, static_vars):
        if sampler in ["StochasticGradientLangevinDynamics",
                                  "GeometricLangevinAlgorithm_1stOrder",
                                  "GeometricLangevinAlgorithm_2ndOrder",
                                  "HamiltonianMonteCarlo",
                                  "BAOAB",
                                  "CovarianceControlledAdaptiveLangevinThermostat"]:
            if sampler == "StochasticGradientLangevinDynamics":
                self.gradients, self.virials, self.noise = \
                    sess.run([
                        static_vars["gradients"],
                        static_vars["virials"],
                        static_vars["noise"]])
            elif sampler == "HamiltonianMonteCarlo":
                self.old_total_energy, self.kinetic_energy, self.momenta, self.gradients, self.virials = \
                    sess.run([
                        static_vars["total_energy"],
                        static_vars["kinetic_energy"],
                        static_vars["momenta"],
                        static_vars["gradients"],
                        static_vars["virials"]])
            else:
                self.kinetic_energy, self.momenta, self.gradients, self.virials, self.noise = \
                    sess.run([
                        static_vars["kinetic_energy"],
                        static_vars["momenta"],
                        static_vars["gradients"],
                        static_vars["virials"],
                        static_vars["noise"]])