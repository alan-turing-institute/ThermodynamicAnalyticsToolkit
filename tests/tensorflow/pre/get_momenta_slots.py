#
#    ThermodynamicAnalyticsToolkit - explore high-dimensional manifold of neural networks
#    Copyright (C) 2018 The University of Edinburgh
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

from TATi.models.model import model
import sys
import tensorflow as tf

FLAGS = model.setup_parameters( \
	batch_data_files=[sys.argv[1]],
	sampler="GeometricLangevinAlgorithm_2ndOrder",
	inverse_temperature=1e3,
	friction_constant=10.,
	step_width=1e-1,
	hidden_dimension=[6,6])
nn = model(FLAGS)
nn.init_network(setup="sample")

for v in tf.trainable_variables():
	print(nn.sampler[0].get_slot(v, "momentum"))

