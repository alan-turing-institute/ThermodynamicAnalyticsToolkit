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

# forgotten assignment
nn = tati(
    input_dimension=2,
    output_dimension=1,
    fix_parameters="output/biases/Variable:0",
    seed=426,
    verbose=2
)

# throws because "test" is not a valid parameter name
nn.dataset = [np.asarray([[0,0]], dtype=np.float32), np.asarray([[1]], dtype=np.int32)]