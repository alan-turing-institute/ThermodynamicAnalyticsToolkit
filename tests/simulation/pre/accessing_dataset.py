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
    batch_size=10,
    seed=426,
)

# store current batch (this will trigger advance)
dataset = nn.dataset

# get loss
print(nn.loss())

# store again, this should not trigger
dataset_same = nn.dataset

# get loss, this should trigger advance
print(nn.loss())

# store again
dataset_different = nn.dataset

for k in range(2):
    assert( np.array_equal( dataset[k], dataset_same[k]))
    assert( not np.array_equal( dataset[k], dataset_different[k]))
    assert( not np.array_equal( dataset_same[k], dataset_different[k]))
