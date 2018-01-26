import math
import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')  # no display
import matplotlib.pyplot as plt
import io


class ClassificationDatasets:
    """
    This class encapsulates all datasets from the TensorFlow playground
    for classification tasks.

    Example:
         dsgen=dataset_generator()
         ds = dsgen.generate(500, 0., dsgen.TWOCIRCLES)

    Attributes:
        TWOCIRCLES: two circles dataset
        SQUARES: four squares dataset
        TWOCLUSTERS: two normally distributed clusters dataset
        SPIRAL: two spiral dataset
    """

    TWOCIRCLES=0
    SQUARES=1
    TWOCLUSTERS=2
    SPIRAL=3

    def __init__(self):
        """ Initializes the class.
        """
        self.xs = []
        self.ys = []
        self.r = 5
        self.func_dict = {
            self.TWOCIRCLES: self.generate_twocircles,
            self.SQUARES: self.generate_squares,
            self.TWOCLUSTERS: self.generate_twoclusters,
            self.SPIRAL: self.generate_spiral,
        }

    def generate(self, dimension, noise, data_type=SPIRAL):
        """ Generates dataset.

        :param dimension: number of items in dataset
        :param noise: noise scale in [0,1]
        :param data_type: which dataset to generate (0,1,2,3)
        :returns: dataset consisting of two-dimensional coordinates and labels
        """
        # clear return instances
        self.xs[:] = []
        self.ys[:] = []

        # call dataset generating function
        if data_type not in self.func_dict:
            raise NotImplementedError("Unknown input data type desired.")
        self.func_dict[data_type](dimension, noise)
        return self.xs, self.ys

    def generate_twocircles(self, dimension, noise):
        """
        Generates two circular distributions with the same
        center but different radii, i.e. they are mostly not
        overlapping.

        :param dimension: number of items in dataset
        :param noise: noise scale in [0,1]
        :returns: dataset consisting of two-dimensional coordinates and labels
        """
        for label in [1,-1]:
            for i in range(int(dimension/2)):
                if label == 1:
                    radius = np.random.uniform(0,self.r*0.5)
                else:
                    radius = np.random.uniform(self.r*0.7, self.r)
                angle = np.random.uniform(0,2*math.pi)
                coords = [radius * math.sin(angle), radius * math.cos(angle)]
                noisecoords = np.random.uniform(-self.r,self.r,2)*noise
                norm = (coords[0]+noisecoords[0])*(coords[0]+noisecoords[0])+(coords[1]+noisecoords[1])*(coords[1]+noisecoords[1])
                self.xs.append(coords)
                self.ys.append([1] if (norm < self.r*self.r*.25) else [-1])
                #print(str(returndata[-1])+" with norm "+str(norm)+" and radius "+str(radius)+": "+str(labels[-1]))
                
    def generate_squares(self, dimension, noise):
        """
        Generates distribution in each of the four quadrants of the two-dimensional domain.

        :param dimension: number of items in dataset
        :param noise: noise scale in [0,1]
        :returns: dataset consisting of two-dimensional coordinates and labels
        """
        '''
        '''
        for i in range(dimension):
            coords = np.random.uniform(-self.r,self.r,2)
            padding = .3
            coords[0] += padding * (1 if (coords[0] > 0) else -1)
            coords[1] += padding * (1 if (coords[1] > 0) else -1)
            noisecoords = np.random.uniform(-self.r,self.r,2)*noise
            self.xs.append(coords)
            self.ys.append([1] if ((coords[0]+noisecoords[0])*(coords[1]+noisecoords[1]) >= 0) else [-1])

    def generate_twoclusters(self, dimension, noise):
        """
        Generates two normal distribution point clouds centered at [2,2] and [-2,-2].

        :param dimension: number of items in dataset
        :param noise: noise scale in [0,1]
        :returns: dataset consisting of two-dimensional coordinates and labels
        """
        variance = 0.5+noise*(3.5*2)
        signs=[1,-1]
        labels=[[1],[-1]]
        for i in range(2):
            for j in range(int(dimension/2)):
                coords = np.random.normal(signs[i]*2,variance,2)
                self.xs.append(coords)
                self.ys.append(labels[i])

    def generate_spiral(self, dimension, noise):
        """
        Generates two spiral-shaped distributions each with a different label.
        This is a standard example for distributions that cannot be sensibly
        distinguished by a linear model, i.e. just a single hidden layer.

        :param dimension: number of items in dataset
        :param noise: noise scale in [0,1]
        :returns: dataset consisting of two-dimensional coordinates and labels
        """
        for deltaT in [0, math.pi]:
            for i in range(int(dimension/2)):
                radius = i/dimension*self.r
                t = 3.5 * i/dimension* 2*math.pi + deltaT
                coords = [radius*math.sin(t)+np.random.uniform(-1,1)*noise,
                          radius*math.cos(t)+np.random.uniform(-1,1)*noise]
                self.xs.append(coords)
                self.ys.append([1] if (deltaT == 0) else [-1])

    @staticmethod
    def get_plot_buf(input_data, input_labels, dimension):
        """ Plots labelled scatter data using matplotlib and returns the created
        PNG as a text buffer.

        This is taken from https://stackoverflow.com/questions/41356093

        :param input_data: data to plot, i.e. x
        :param input_labels: labels to color input coordinates by, i.e. y
        :param dimension: dimension of the dataset
        """
        plt.figure()
        plt.scatter([val[0] for val in input_data], [val[1] for val in input_data],
                    s=dimension,
                    c=[('r' if (label[0] > 0.) else 'b') for label in input_labels])
        # c=input_labels)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf

    def add_graphing_train(self):
        """ Adds nodes for visualization of the dataset with predicted labels.

        This can be used to visualize the current prediction of the dataset
        and thereby to assess the progress of the training. The images are
        PNGs and are stored to files, see :method:`classification_datasets.graph_truth`
        TensorBoard visualizes these under `images` if found in its logdir.

        :return: placeholder for feeding in string encoded PNG image, summary
            node for writing image to
        """
        # print("Adding graphing nodes")
        plot_buf_test = tf.placeholder(tf.string)
        image_test = tf.image.decode_png(plot_buf_test, channels=4)
        image_test = tf.expand_dims(image_test, 0) # make it batched
        plot_image_summary_test = tf.summary.image('test', image_test, max_outputs=1)
        return plot_buf_test, plot_image_summary_test

    def graph_truth(self, sess, data, labels, sample_size, log_writer):
        """

        :param sess: TensorFlow session
        :param data: input of dataset, i.e. x
        :param labels: labels of dataset, i.e. y
        :param sample_size: size of the dataset
        :param log_writer: writer node for writing the images to
        :return: plot_image_summary_truth
        """
        plot_buf_truth = tf.placeholder(tf.string)
        image_truth = tf.image.decode_png(plot_buf_truth, channels=4)
        image_truth = tf.expand_dims(image_truth, 0) # make it batched
        plot_image_summary_truth = tf.summary.image('truth', image_truth, max_outputs=1)
        plot_buf = self.get_plot_buf(data, labels, sample_size)
        plot_image_summary_ = sess.run(
            plot_image_summary_truth,
            feed_dict={plot_buf_truth: plot_buf.getvalue()})
        log_writer.add_summary(plot_image_summary_, global_step=0)

