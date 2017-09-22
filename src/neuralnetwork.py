import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg') # no display
import matplotlib.pyplot as plt
import io

class neuralnetwork:
    ''' This class encapsulates the construction of the neural network.
    '''
    
    summary_nodes = {}
    
    def get(self, keyname):
        ''' This is just a short hand to access the summary nodes dict.
        '''
        return self.summary_nodes[keyname]
    
    def create(self, input_layer,
               input_dimension, layer_dimensions, output_dimension,
               learning_rate):
        ''' Create function for the actual net in TensorFlow where
        full summary nodes are added.
        '''
        self.summary_nodes.clear()
    
        y_ = tf.placeholder(tf.float32, [None, output_dimension], name='y-input')
        print("y_ is "+str(y_.get_shape()))
        self.summary_nodes['y_'] = y_

        keep_prob = tf.placeholder(tf.float32)
        self.summary_nodes['keep_prob'] = keep_prob
        with tf.name_scope('dropout'):
            tf.summary.scalar('dropout_keep_probability', keep_prob)

        layer_no=1
        current_hidden = None
        current_dropped = input_layer
        in_dimension = None
        out_dimension = input_dimension
        for i in range(len(layer_dimensions)):
            number=str(i+1)
            in_dimension = out_dimension
            out_dimension = layer_dimensions[i]
            layer_name = "layer"+number
            current_hidden = self.nn_layer(current_dropped, in_dimension, out_dimension, layer_name)
            print(layer_name+" is "+str(current_hidden.get_shape()))

            with tf.name_scope('dropout'):
                current_dropped = tf.nn.dropout(current_hidden, keep_prob)
            print("dropped"+number+" is "+str(current_dropped.get_shape()))
        
        # Do not apply softmax activation yet, see below.
        y = self.nn_layer(current_dropped, out_dimension, output_dimension, 'output', act=tf.nn.tanh)
        self.summary_nodes['y'] = y
        print("y is "+str(y.get_shape()))

        print ("Creating summaries")
        with tf.name_scope('loss'):
            with tf.name_scope('total'):
                loss = tf.losses.mean_squared_error(labels=y_, predictions=y)
                self.summary_nodes['loss'] = loss
        tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
                    loss)
        self.summary_nodes['train_step'] = train_step

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.sign(y), tf.sign(y_))
                self.summary_nodes['correct_prediction'] = correct_prediction
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                self.summary_nodes['accuracy'] = accuracy
        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to
        # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
        merged = tf.summary.merge_all()
        self.summary_nodes['merged'] = merged
        
        # return global truth and dropout probability nodes as
        # they need to be filled in
        return y_, keep_prob
        
    def add_writers(self, sess, log_dir):
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        self.summary_nodes["train_writer"] = train_writer
        test_writer = tf.summary.FileWriter(log_dir + '/test')
        self.summary_nodes["test_writer"] = test_writer
        
    def init_graph(self, sess):
        print ("Initializing global variables")
        sess.run(tf.global_variables_initializer())

    def add_graphing_train(self):
        print("Adding graphing nodes")       
        plot_buf_test = tf.placeholder(tf.string)
        self.summary_nodes["plot_buf_test"] = plot_buf_test
        image_test = tf.image.decode_png(plot_buf_test, channels=4)
        image_test = tf.expand_dims(image_test, 0) # make it batched
        plot_image_summary_test = tf.summary.image('test', image_test, max_outputs=1)
        self.summary_nodes["plot_image_summary_test"] = plot_image_summary_test

    def graph_truth(self, sess, data, labels, sample_size):
        plot_buf_truth = tf.placeholder(tf.string)
        image_truth = tf.image.decode_png(plot_buf_truth, channels=4)
        image_truth = tf.expand_dims(image_truth, 0) # make it batched
        plot_image_summary_truth = tf.summary.image('truth', image_truth, max_outputs=1)
        self.summary_nodes["plot_image_summary_truth"] = plot_image_summary_truth
        plot_buf = self.get_plot_buf(data, labels, sample_size)
        plot_image_summary_ = sess.run(
            plot_image_summary_truth,
            feed_dict={plot_buf_truth: plot_buf.getvalue()})
        test_writer = self.get("test_writer")
        test_writer.add_summary(plot_image_summary_, global_step=0)

        
    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.random_uniform(shape, minval=-0.5, maxval=0.5)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def nn_layer(self,
                 input_tensor, input_dim, output_dim,
                 layer_name, act=tf.nn.relu):
        """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        print("Creating nn layer %s with %d, %d" % (layer_name, input_dim, output_dim))
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable([input_dim, output_dim])
                self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                self.variable_summaries(biases)
            with tf.name_scope('Wx_plus_b'):
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)
            if act == None:
                return preactivate
            else:
                activations = act(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations

    def get_plot_buf(self, input_data, input_labels, dimension):
        ''' Plots labelled scatter data using matplotlib and returns the created
        PNG as a text buffer
        
        This is taken from https://stackoverflow.com/questions/41356093
        '''
        plt.figure()
        plt.scatter([val[0] for val in input_data], [val[1] for val in input_data],
                    s=dimension,
                    c=[('r' if (label[0] > 0.) else 'b') for label in input_labels])
                    #c=input_labels)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf
