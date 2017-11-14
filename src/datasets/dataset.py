import math
import numpy as np


class Dataset:
    """ This class contains a set of samples for training.
    It provides function to split the dataset into batches.
    It mimics the essential stuff from TensorFlow's Dataset
    class.
    
    To use the class as instance ds:
    1. First provide data by calling: ds.init(xs,ys)
    2. Next, obtain batches by: bxs,bys = ds.next_batch(10)
    3. Check, whether you have obtained all batches ones: ds.epochFinished()
    4. Then, you have done one training step. Continue at step 2.

    """
    
    batch_start = 0
    test_train_ratio = 0.5
    slice_index = 0

    def __init__(self, xs, ys):
        """ Sets up the dataset with a given array of data and labels.

        This splits the dataset into two parts: one for training and
        the other for testing. Moreover, it shuffles the dataset to
        avoid clusters of same labels.

        :param xs: input data
        :param ys: label data for input
        """
        self.xs = xs
        self.ys = ys
        self.set_test_train_ratio(0.5)
        self.slice_index = math.floor(len(self.xs) * self.test_train_ratio)
        # shuffle once to prevent and undistributed testset
        self.shuffle()

    def set_test_train_ratio(self, ratio):
        """ Sets the test/train ratio of the dataset, i.e. where to split.

        This basically set the slice_index.

        :param ratio: ratio in [0,1]
        """
        self.test_train_ratio = ratio
        self.slice_index = math.floor(len(self.xs) * self.test_train_ratio)

    def get_testset(self):
        """ Returns the current testset set for this epoch as a whole.

        :return: tuple of testset input and testset labels.
        """
        test_xs, test_ys = self.xs[:self.slice_index], self.ys[:self.slice_index]
        assert not any(item is None for item in [test_xs, test_ys])
        # print("Testset is x: "+str(test_xs[0:5])+", y: "+str(test_ys[0:5]))
        return test_xs, test_ys

    def next_batch(self, batch_size):
        """ Returns next batch from internally stored samples.

        :param batch_size: size of batch that splits up datasets
        :return: tuple of input and labels for this batch
        """
        if self.epochStarted():
            # shuffle on start of epoch
            self.shuffle()
        if self.batch_start + batch_size >= self.slice_index:
            # return rest
            batch_xs = self.xs[self.batch_start:self.slice_index]
            batch_ys = self.ys[self.batch_start:self.slice_index]
            remaining_size = batch_size-(self.slice_index-self.batch_start)
            if remaining_size != 0:
                # reshuffle like on epoch restart
                self.shuffle()
                # note that this copies the array and is not inplace!
                batch_xs = np.append(batch_xs, self.xs[0:remaining_size])
                batch_ys = np.append(batch_ys, self.ys[0:remaining_size])
            self.batch_start = remaining_size
        else:
            # return full batch
            batch_end = self.batch_start+batch_size
            batch_xs = self.xs[self.batch_start:batch_end]
            batch_ys = self.ys[self.batch_start:batch_end]
            self.batch_start = batch_end
        return batch_xs, batch_ys
    
    def epochFinished(self):
        """ Predicate for epoch is at end, i.e. batch ran through dataset.

        :return: True - end of epoch, False - else
        """
        return self.batch_start == self.slice_index
    
    def epochStarted(self):
        """ Predicate for when we are at the very first batch of the dataset.

        :return: True - are at first batch, False - else
        """
        return self.batch_start == 0

    def resetEpoch(self):
        """ Resets the batch to start at the datasets's beginning.
        """
        self.batch_start = 0

    def clear(self):
        """ Resets the internal state.
        """
        self.xs[:] = []
        self.ys[:] = []
        self.epochs = 0
        self.batch_start = 0

    def shuffle(self):
        """ Shuffle data set.
        """
        randomize = np.arange(len(self.xs))
        np.random.shuffle(randomize)
        self.xs = np.array(self.xs)[randomize]
        self.ys = np.array(self.ys)[randomize]
