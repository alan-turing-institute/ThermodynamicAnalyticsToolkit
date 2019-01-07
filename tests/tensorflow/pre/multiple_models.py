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

import tensorflow as tf

from TATi.models.model import model

import numpy as np
import sys

FLAGS = model.setup_parameters(
    batch_data_files=[sys.argv[1]],
    batch_size=500,
    max_steps=1000,
    output_activation="linear",
    sampler="GeometricLangevinAlgorithm_2ndOrder",
    seed=426,
    step_width=1e-2
)
print(FLAGS)

FLAGS2 = FLAGS
FLAGS2.seed = 427
print(FLAGS)
print(FLAGS2)


nn1 = model(FLAGS)
nn1.init_network(None, setup="sample")
nn1.init_input_pipeline()
run_info, trajectory, averages = nn1.sample(return_run_info=True, \
  return_trajectories=True, return_averages=True)

print("Sample results")
print(np.asarray(run_info[0:10]))
print(np.asarray(trajectory[0:10]))
print(np.asarray(averages[0:10]))

tf.reset_default_graph()

nn2 = model(FLAGS)
nn2.init_network(None, setup="sample")
nn2.init_input_pipeline()
run_info, trajectory, averages = nn2.sample(return_run_info=True, \
  return_trajectories=True, return_averages=True)


