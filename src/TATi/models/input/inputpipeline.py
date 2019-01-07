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



class InputPipeline(object):
    """ This class defines the interface for an input pipeline as uses by
    model.sample() and model.train() to obtain batches of the dataset.

    """

    NUM_PARALLEL_CALLS = 10 # number of parallel threads for input pipeline parsing

    def __init__(self, filenames,
                 batch_size, dimension, max_steps,
                 input_dimension, output_dimension,
                 shuffle, seed):
        pass

    def next_batch(self, session):
        ''' This returns the next batch of features and labels.

        :param session: session object as input might be retrieved through the
                computational graph
        :return: pack of feature and label array
        '''
        assert( False )

    def epochFinished(self):
        ''' This checks whether the epoch is done, i.e. whether the dataset
        is depleted and needs to be reset.

        :return: True - epoch is done, False - else
        '''
        assert( False )

    def reset(self, session):
        ''' This resets the dataset such that a new epoch of training or
        sampling may commence.

        :param session: session object as input might be retrieved through the
                computational graph
        '''
        assert( False )

    def shuffle(self, seed):
        ''' This shuffles the dataset.

        :param seed: seed used for random shuffle to allow reproducible runs
        '''
        assert( False )
