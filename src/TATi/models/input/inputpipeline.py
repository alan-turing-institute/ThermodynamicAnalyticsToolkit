

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
