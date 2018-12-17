import logging
import numpy as np
import os
import sys

from TATi.analysis.parsedrunfile import ParsedRunfile
from TATi.analysis.parsedtrajectory import ParsedTrajectory
from TATi.analysis.averageenergieswriter import AverageEnergiesWriter
from TATi.analysis.averagetrajectorywriter import AverageTrajectoryWriter
from TATi.analysis.diffusionmap import DiffusionMap
from TATi.analysis.freeenergy import FreeEnergy

class AnalyserModules(object):
    """ This class contains all analyser modules with their dependencies in
    such a way that a single string, e.g., "diffusion_map" (and information
    in \a FLAGS) is sufficient to execute this analysis on a trajectory.

    To this end, the modules are setup hierarchically. More elaborate ones
    depend on the result of simpler ones. For example, before computing
    a diffusion map we need to parse the trajectory. Free energy computations
    depend on a diffusion map and so on. Therefore, all modules are sorted
    in a dependency graph and store their results in a dict.

    """

    # available analysis modules, i.e.
    # for each module "foobar", there needs to be a function analyse_foobar()
    # and the other way round: all functions analyse_...() need to be enlisted
    # here to be executable by TATiAnalyser.
    analysis_modules = ("parse_run_file", "parse_trajectory_file", "average_energies", "average_trajectory",
                        "diffusion_map", "free_energy_levelsets", "free_energy_histograms")

    # list all dependencies between the different analysis modules, i.e. what
    # needs to be done before the module itself is run. Note that subdependencies
    # (chains) are adhered, i.e. only list direct requirements.
    analysis_dependencies = {
        "parse_run_file": [],
        "parse_trajectory_file": [],
        "average_energies": ["parse_run_file"],
        "average_trajectory": ["parse_trajectory_file"],
        "diffusion_map": ["parse_trajectory_file"],
        "free_energy_levelsets": ["diffusion_map"],
        "free_energy_histograms": ["diffusion_map"],
    }

    # stores all results obtained through a specific analysis module, i.e.
    # caching results for dependent stages
    analysis_storage = {
        "parse_run_file": [None],
        "parse_trajectory_file": [None],
        "average_energies": [None],
        "average_trajectory": [None],
        "diffusion_map": [None],
        "free_energy_levelsets": [None],
        "free_energy_histograms": [None],
    }

    def __init__(self, FLAGS, output_width, output_precision):
        self.FLAGS = FLAGS
        self.output_width= output_width
        self.output_precision= output_precision

    def _get_all_dependencies(self, name):
        retlist = []
        list_to_check = [name]
        while len(list_to_check) != 0:
            # pop item from check list
            current_module = list_to_check[0]
            del list_to_check[0]
            # look at all dependencies
            if current_module in self.analysis_dependencies.keys():
                deplist = self.analysis_dependencies[current_module]
                for dep in deplist:
                    if dep not in retlist:
                        retlist.append(dep)
                        if dep not in list_to_check:
                            list_to_check.append(dep)
            else:
                logging.warning("Cannot find "+name+" in dependencies.")
        return retlist

    def _get_all_required_modules(self, list_of_modules):
        retlist = []
        for current_module in list_of_modules:
            retlist.extend(self._get_all_dependencies(current_module))
        return list(set(retlist))

    def _sort_modules_for_execution(self, list_of_modules):
        sorted_list = []
        # we need to sort the list according to the dependencies
        temp_list = list(list_of_modules)
        while len(temp_list) != 0:
            # pop first item
            current_module = temp_list[0]
            del temp_list[0]

            # check dependencies (or all fulfilled by sorted_list)
            deplist = self._get_all_dependencies(current_module)
            status = True
            for dep in deplist:
                if dep not in sorted_list:
                    status = False
                    break

            # if yes, push to sorted_list
            if status:
                sorted_list.append(current_module)
            # if not, push to end of temp_list again
            else:
                temp_list.append(current_module)
        return sorted_list

    def execute(self, list_of_modules):
        sorted_list = self._sort_modules_for_execution(
            self._get_all_required_modules(list_of_modules)+list_of_modules)
        logging.debug("Execution sequence: "+str(sorted_list))
        # then execute each stage
        for module in sorted_list:
            logging.info("Executing analysis module "+module)
            getattr(self, "analyse_"+module)()

    def get_stage_results(self, name):
        if self.analysis_storage[name] is not None:
            return self.analysis_storage[name]
        else:
            logging.critical("Results from dependent stage "+name+" not present!?")
            sys.exit(127)

    def analyse_parse_run_file(self):
        runfile = ParsedRunfile(self.FLAGS.run_file, self.FLAGS.every_nth)
        self.analysis_storage["parse_run_file"][0] = runfile

        # write results
        if self.FLAGS.run_file is not None:
            print("Run file is " + str(self.FLAGS.run_file))
            # load run file
            if not runfile.add_drop_burnin(self.FLAGS.drop_burnin):
                sys.stderr.write("self.FLAGS.drop_burnin is too large, no data points left.")
                sys.exit(1)

            print("%d steps after dropping burn in." % (runfile.number_steps()))
            print("%lg average and %lg variance in runfile.get_loss()." % \
                  (np.average(runfile.get_loss()), runfile.get_loss().var()))

    def analyse_parse_trajectory_file(self):
        print("Trajectory file is " + str(self.FLAGS.trajectory_file))
        trajectory = ParsedTrajectory(self.FLAGS.trajectory_file, self.FLAGS.every_nth)
        self.analysis_storage["parse_trajectory_file"][0] = trajectory

        # write results
        if self.FLAGS.trajectory_file is not None:
            # load trajectory file
            if not trajectory.add_drop_burnin(self.FLAGS.drop_burnin):
                sys.stderr.write("self.FLAGS.drop_burnin is too large, no data points left.")
                sys.exit(1)

    def analyse_average_energies(self):
        runfile = self.analysis_storage["parse_run_file"][0]
        if self.FLAGS.average_run_file is not None:
            avg_writer = AverageEnergiesWriter(runfile, self.FLAGS.steps)
            avg_writer.write(self.FLAGS.average_run_file)

    def analyse_average_trajectory(self):
        trajectory = self.get_stage_results("parse_trajectory_file")[0]
        if self.FLAGS.average_trajectory_file is not None:
            averagewriter = AverageTrajectoryWriter(trajectory.get_trajectory())
            averagewriter.write(self.FLAGS.average_trajectory_file)

    def analyse_diffusion_map(self):
        # compute diffusion map and write to file
        trajectory = self.get_stage_results("parse_trajectory_file")[0]
        dmap = DiffusionMap.from_parsedtrajectory(trajectory)
        if self.FLAGS.inverse_temperature is None or self.FLAGS.number_of_eigenvalues is None:
            print("Require both inverse_temperature and number_of_eigenvalues.")
            sys.exit(255)
        if not dmap.compute(number_eigenvalues=self.FLAGS.number_of_eigenvalues,
                            inverse_temperature=self.FLAGS.inverse_temperature,
                            diffusion_map_method=self.FLAGS.diffusion_map_method,
                            use_reweighting=self.FLAGS.use_reweighting):
            logging.warning("Diffusion Map computation failed, not computing landmarks.")
            self.FLAGS.landmarks = 0
        self.analysis_storage["diffusion_map"][0] = dmap

        # write results
        if self.FLAGS.diffusion_map_file is not None or self.FLAGS.diffusion_matrix_file is not None \
                or self.FLAGS.landmarks is not None:


            if self.FLAGS.diffusion_map_file is not None and not os.path.isfile(self.FLAGS.diffusion_map_file):
                print("Eigenvalues are " + str(dmap.values))
                dmap.write_values_to_csv(self.FLAGS.diffusion_map_file,
                                         self.output_precision, self.output_width)

            if self.FLAGS.diffusion_matrix_file is not None and not os.path.isfile(self.FLAGS.diffusion_matrix_file):
                dmap.write_vectors_to_csv(self.FLAGS.diffusion_matrix_file,
                                          self.output_precision, self.output_width)

    def analyse_free_energy_levelsets(self):
        trajectory = self.get_stage_results("parse_trajectory_file")[0]
        dmap = self.get_stage_results("diffusion_map")[0]
        freeenergy = FreeEnergy(trajectory, dmap)
        freeEnergies, levelsets = freeenergy.compute_on_levelsets( \
            num_landmarks=self.FLAGS.landmarks)
        self.analysis_storage["free_energy_levelsets"] = [freeEnergies, levelsets]

        landmarks = dmap.compute_landmarks(self.FLAGS.landmarks)
        self.analysis_storage["free_energy_levelsets"].append(landmarks)

        # write results
        if self.FLAGS.landmark_prefix is not None and self.FLAGS.landmarks > 0:
            steps = trajectory.get_steps()
            for ev_index in range(np.shape(dmap.vectors)[1]):
                landmark_filename = self.FLAGS.landmark_prefix + "-ev_" + str(ev_index + 1) + ".csv"
                freeenergy.write_levelsets_to_csv( \
                    steps=steps,
                    ev_index=ev_index,
                    landmarks=landmarks[ev_index],
                    freeEnergies=freeEnergies[ev_index],
                    landmark_filename=landmark_filename,
                    output_width=self.output_width, output_precision=self.output_precision)

    def analyse_free_energy_histograms(self):
        trajectory = self.get_stage_results("parse_trajectory_file")[0]
        dmap = self.get_stage_results("diffusion_map")[0]
        freeenergy = FreeEnergy(trajectory, dmap)
        freeEnergies, HistogramBins = freeenergy.compute_by_histogramming( \
            num_bins=self.FLAGS.landmarks)
        self.analysis_storage["free_energy_histograms"] = [freeEnergies, HistogramBins]
        if self.FLAGS.free_energy_prefix is not None:
            for ev_index in range(len(freeEnergies)):
                filename = self.FLAGS.free_energy_prefix + "-ev_" + str(ev_index + 1) + ".csv"
                freeenergy.write_histograms_to_csv(freeEnergies[ev_index], HistogramBins[ev_index], filename,
                                                   output_width=self.output_width,
                                                   output_precision=self.output_precision)
