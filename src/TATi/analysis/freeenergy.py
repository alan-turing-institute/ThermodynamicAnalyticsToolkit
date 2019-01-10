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

import logging
import numpy as np

import TATi.diffusion_maps.diffusionmap as dm

from TATi.common import setup_csv_file

class FreeEnergy(object):
    """This class wraps the capability to perform a free energy calculation.
    
    The concept of free enery allows to make a relative comparison between
    different minima basins by taking the proportions of all minima basins
    into account.
    
    Note:
        Both analysis variants - levelsets and histograms - do not yet give
        results that completely coincide. The concept of free energy still
        needs to be properly defined for neural network loss manifolds.
    
    Warning:
    
        THIS IS STILL EXPERIMENTAL.

    Args:

    Returns:

    """

    def __init__(self, trajectory, loss, dmap_vectors, dmap_q):
        self.trajectory = trajectory
        self.loss = loss
        self.dmap_vectors = dmap_vectors
        self.dmap_q = dmap_q

    def __init__(self, parsedtrajectory, dmap):
        self.trajectory = parsedtrajectory.get_trajectory()
        self.loss = parsedtrajectory.get_loss()
        self.dmap_vectors = dmap.vectors
        self.dmap_q = dmap.q

    def compute_on_levelsets(self, num_landmarks):
        logging.warning("Free energy analysis using levelsets is not yet tested and probably does not give meaningful results, yet.")
        print("Computing free energy")
        #compute levelsets

        freeEnergies = []
        Levelsets = []
        for vindex in range(np.shape(self.dmap_vectors)[1]):
            V1 = self.dmap_vectors[:, vindex]
            levelsets, _ = dm.get_levelsets(self.trajectory, num_landmarks,
                                            self.dmap_q, V1)

            K=len(levelsets)
            freeEnergy=np.zeros(K)
            h=np.zeros(K)

            for k in range(0,K):
                h[k] = len(levelsets[k])

            for k in range(0,K):
                freeEnergy[k] = -np.log(h[k]/sum(h))

            freeEnergies.append(freeEnergy)
            Levelsets.append(levelsets[0])

        print("freeEnergies: " + str(freeEnergies))
        print("Levelsets: " + str(Levelsets))
        return freeEnergies, Levelsets

    def compute_by_histogramming(self, num_bins):
        """Calculate free energy by histogramming the eigenvector.

        Args:
          num_bins: 

        Returns:

        """
        logging.warning("Free energy analysis using histograms is not yet tested and probably does not give meaningful results, yet.")
        mlist = []
        for ev_index in range(np.shape(self.dmap_vectors)[1]):
            mlist.append(self._compute_free_energy_using_histograms(
                radius=self.dmap_vectors[:, ev_index],
                nrbins=num_bins))
        free_energy, bins = zip(*mlist)
        return free_energy, bins

    def write_levelsets_to_csv(self, steps, ev_index, landmarks, freeEnergies, landmark_filename, \
                               output_width, output_precision):
        header = ["landmark", "loss", "kernel_diff"]
        for i in range(np.shape(self.trajectory)[1]):
            header.append("dof_" + str(i))
        header.append("free_energy")
        header.append("ev_" + str(ev_index + 1))
        csv_writer, csv_file = setup_csv_file(landmark_filename, header)
        for k in range(np.shape(landmarks)[0]):
            i = landmarks[k]
            csv_writer.writerow([steps[i, 0]]
                                + ['{:{width}.{precision}e}'.format(self.loss[i, 0], \
                                                                    width=output_width, precision=output_precision)] \
                                + ['{:{width}.{precision}e}'.format(self.dmap_q[i, 0], \
                                                                    width=output_width, precision=output_precision)] \
                                + ['{:{width}.{precision}e}'.format(x, \
                                                                    width=output_width, precision=output_precision)
                                   for x in np.asarray(self.trajectory[i, :])] \
                                + ['{:{width}.{precision}e}'.format(freeEnergies[k], \
                                                                    width=output_width, precision=output_precision)] \
                                + ['{:{width}.{precision}e}'.format(np.real(self.dmap_vectors[i, ev_index]), \
                                                                    width=output_width,
                                                                    precision=output_precision)])
        csv_file.close()

    @staticmethod
    def _compute_free_energy_using_histograms(radius, weights=None, nrbins=100, kBT=1):

        free_energy, edges=np.histogram(radius, bins=nrbins, weights = weights, normed=True)
        #free_energy+=0.0001
        free_energy= - np.log(free_energy)

        logging.debug(edges.shape)

        return free_energy, edges[:-1]

    @staticmethod
    def write_histograms_to_csv(freeEnergies, HistogramBins, filename, \
                                output_width, output_precision):
        header = ["bin", "free_energy"]
        csv_writer, csv_file = setup_csv_file(filename, header)
        for k in range(len(freeEnergies)):
            csv_writer.writerow(
                ['{:{width}.{precision}e}'.format(HistogramBins[k], \
                                                  width=output_width, precision=output_precision)] \
                + ['{:{width}.{precision}e}'.format(freeEnergies[k], \
                                                    width=output_width, precision=output_precision)])
        csv_file.close()
