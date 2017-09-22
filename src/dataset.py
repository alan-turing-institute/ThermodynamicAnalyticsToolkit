import math
import numpy as np

class dataset:
    ''' This class contains a set of samples for training.
    It provides function to split the dataset into batches.
    It mimicks the essential stuff from TensorFlow's Dataset
    class.
    
    To use the class as instance ds:
    1. First provide data by calling: ds.init(xs,ys)
    2. Next, obtain batches by: bxs,bys = ds.next_batch(10)
    3. Check, whether you have obtained all batches ones: ds.epochFinished()
    4. Then, you have done one training step. Continue at step 2.
    '''
    
    xs = []
    ys = []
    batch_start = 0
    testtrain_ratio = 0.5
    sliceindex = 0

    def __init__(self, xs, ys):
        ''' Sets up the dataset with a given array of data and labels.
        '''
        self.xs[:] = xs
        self.ys[:] = ys
        self.sliceindex = math.floor(len(self.xs)*self.testtrain_ratio)
        # shuffle once to prevent and undistributed testset
        self.shuffle()

    def get_testset(self):
        ''' Returns the current teset set for this epoch as a whole.
        '''
        return self.xs[:self.sliceindex], self.ys[:self.sliceindex]

    def next_batch(self, batch_size):
        ''' Returns next batch from internally stored samples.
        '''
        if self.epochStarted():
            # shuffle on start of epoch
            self.shuffle()
        if self.batch_start + batch_size >= self.sliceindex:
            # return rest
            batch_xs = self.xs[self.batch_start:self.sliceindex]
            batch_ys = self.ys[self.batch_start:self.sliceindex]
            self.batch_start = self.sliceindex
        else:
            # return full batch
            batch_end = self.batch_start+batch_size
            batch_xs = self.xs[self.batch_start:batch_end]
            batch_ys = self.ys[self.batch_start:batch_end]
            self.batch_start = batch_end
        return batch_xs, batch_ys
    
    def epochFinished(self):
        return self.batch_start == self.sliceindex
    
    def epochStarted(self):
        return self.batch_start == 0

    def resetEpoch(self):
            self.batch_start = 0

    def clear(self):
        ''' Resets the internal state.
        '''
        self.xs[:] = []
        self.ys[:] = []
        self.epochs = 0
        self.batch_start = 0

    def shuffle(self):
        ''' shuffle data set.
        '''
        randomize = np.arange(len(self.xs))
        np.random.shuffle(randomize)
        self.xs = np.array(self.xs)[randomize]
        self.ys = np.array(self.ys)[randomize]
        