from DataDrivenSampler.models.input.datasetpipeline import DatasetPipeline

import tensorflow as tf
from math import ceil
import sys

class InMemoryPipeline(DatasetPipeline):
    """ This specific input pipeline uses a numpy array as an in-memory dataset
    feeds it directly to the feed_dict of the tensorflow session using placeholders.
    """

    def __init__(self, dataset,
                 batch_size, max_steps,
                 shuffle, seed):
        '''

        :param dataset: dataset array with [features, labels]
        :param batch_size: number of datums to return
        :param max_steps: maximum number of steps for optimizing/sampling
        :param shuffle: whether to shuffle dataset initially or not
        :param seed: seed used for random shuffle to allow reproducible runs
        '''
        self.features = dataset[0]
        self.labels = dataset[1]
        assert( self.features.shape[0] == self.labels.shape[0] )
        dimension = self.features.shape[0]

        self.features_placeholder = tf.placeholder(self.features.dtype, self.features.shape)
        self.labels_placeholder = tf.placeholder(self.labels.dtype, self.labels.shape)

        self.dataset = tf.data.Dataset.from_tensor_slices((self.features_placeholder, self.labels_placeholder))

        if shuffle:
            self.dataset = self.dataset.shuffle(seed=seed)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.repeat(ceil(max_steps*batch_size/dimension))
        print(self.dataset.output_shapes)
        print(self.dataset.output_types)

        self.iterator = self.dataset.make_initializable_iterator()
        self.batch_next = self.iterator.get_next()

    def reset(self, session):
        ''' This resets the dataset such that a new epoch of training or
        sampling may commence.

        :param session: session object as input might be retrieved through the
                computational graph
        '''
        session.run(self.iterator.initializer,
                    feed_dict={self.features_placeholder: self.features,
                               self.labels_placeholder: self.labels})

