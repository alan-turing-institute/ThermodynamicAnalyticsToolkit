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

import numpy as np
import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    number_walkers=2,
)

# check zeroing of all parameters
nn.parameters = np.zeros((nn.num_parameters()))
for k in range(nn._options.number_walkers):
    for i in range(nn.num_parameters()):
        assert( nn.parameters[k][i] == 0. )

# check single walker, multiple parameters access
nn.parameters = np.zeros((nn.num_parameters()))
nn.parameters[1] = np.ones((nn.num_parameters()))
assert( nn.parameters[1][0] == 1. )
assert( nn.parameters[1][1] == 1. )
assert( nn.parameters[0][0] == 0. )
assert( nn.parameters[0][1] == 0. )

# check multiple walker, multiple parameters access
nn.parameters = np.zeros((nn.num_parameters()))
nn.parameters = np.ones((nn.num_parameters()))
assert( nn.parameters[1][0] == 1. )
assert( nn.parameters[1][1] == 1. )
assert( nn.parameters[0][0] == 1. )
assert( nn.parameters[0][1] == 1. )
