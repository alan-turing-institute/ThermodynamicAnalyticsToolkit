#!/usr/bion/pyton3
#
# This a command-line version of some of the features found under
# http://playground.tensorflow.org/
# aimed at comparing steepest descent optimizer with a bayesian
# sampling approach.
#
# (C) Frederik Heber 2017-09-18

import argparse, os, sys
import random as rand
import math
import tensorflow as tf
import numpy as np

import matplotlib as mpl
mpl.use('Agg') # no display
import matplotlib.pyplot as plt
import io


FLAGS = None

TWOCIRCLES=0
SQUARES=1
TWOCLUSTERS=2
SPIRAL=3

def generate_input_data(dimension, noise, data_type=SPIRAL):
    '''
    Generates the spiral input data where
    data_type decides which type to generate.
    All data resides in the domain [-6,6]^2.
    '''
    returndata = []
    labels = []
    r = 5
    if data_type == TWOCIRCLES:
        for label in [1,-1]:
            for i in range(int(dimension/2)):
                if label == 1:
                    radius = np.random.uniform(0,r*0.5)
                else:
                    radius = np.random.uniform(r*0.7, r)
                angle = np.random.uniform(0,2*math.pi)
                coords = [radius * math.sin(angle), radius * math.cos(angle)]
                noisecoords = np.random.uniform(-r,r,2)*noise
                norm = (coords[0]+noisecoords[0])*(coords[0]+noisecoords[0])+(coords[1]+noisecoords[1])*(coords[1]+noisecoords[1])
                returndata.append(coords)
                labels.append([1, 0] if (norm < r*r*.25) else [0, 1])
                #print(str(returndata[-1])+" with norm "+str(norm)+" and radius "+str(radius)+": "+str(labels[-1]))
    elif data_type == SQUARES:
        for i in range(dimension):
            coords = np.random.uniform(-r,r,2)
            padding = .3
            coords[0] += padding * (1 if (coords[0] > 0) else -1)
            coords[1] += padding * (1 if (coords[1] > 0) else -1)
            noisecoords = np.random.uniform(-r,r,2)*noise
            returndata.append(coords)
            labels.append([1, 0] if ((coords[0]+noisecoords[0])*(coords[1]+noisecoords[1]) >= 0) else [0, 1])
    elif data_type == TWOCLUSTERS:
        variance = 0.5+noise*(3.5*2)
        signs=[1,-1]
        labels=[[1,0],[0,1]]
        for i in range(2):
            for j in range(int(dimension/2)):
                coords = np.random.normal(signs[i]*2,variance,2)
                returndata.append(coords)
                labels.append(labels[i])
    elif data_type == SPIRAL:
        for deltaT in [0, math.pi]:
            for i in range(int(dimension/2)):
                radius = i/dimension*r
                t = 3.5 * i/dimension* 2*math.pi + deltaT
                coords = [radius*math.sin(t)+np.random.uniform(-1,1)*noise,
                          radius*math.cos(t)+np.random.uniform(-1,1)*noise]
                returndata.append(coords)
                labels.append([1, 0] if (deltaT == 0) else [0, 1])
    else:
        print("Unknown input data type desired.")
    randomize = np.arange(len(returndata))
    np.random.shuffle(randomize)
    return [np.array(returndata)[randomize], np.array(labels)[randomize]]

# We can't initialize these variables to 0 - the network will get stuck.
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
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

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
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
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        if act == None:
            return preactivate
        else:
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
            return activations

def get_plot_buf(input_data, input_labels):
    ''' Plots labelled scatter data using matplotlib and returns the created
    PNG as a text buffer
    '''
    plt.figure()
    plt.scatter([val[0] for val in input_data], [val[1] for val in input_data],
                s=FLAGS.dimension,
                c=[('r' if (label[0] >= .9) else 'b') for label in input_labels])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def main(_):
    print("Generating input data")
    [input_data, input_labels] = generate_input_data(
        dimension=FLAGS.dimension,
        noise=FLAGS.noise,
        data_type=FLAGS.data_type)

    sess = tf.InteractiveSession()
        
    print("Constructing neural network")
    input_dimension = 2
    hidden_dimension = 4

    # Input placeholders
    with tf.name_scope('input'):
        xinput = tf.placeholder(tf.float32, [None, input_dimension], name='x-input')
        #print("xinput is "+str(xinput.get_shape()))
        y_ = tf.placeholder(tf.float32, [None, 2], name='y-input')
        #print("y_ is "+str(y_.get_shape()))

        # pick from the various available input columns
        x1 = xinput[:,0]
        #print("x1 is "+str(x1.get_shape()))
        x2 = xinput[:,1]
        #print("x2 is "+str(x2.get_shape()))
        x12 = x1*x1
        x22 = x2*x2
        sinx1 = tf.sin(x1)
        sinx2 = tf.sin(x2)
        input_list = [1, 2]
        picked_list = []
        if 1 in input_list:
            picked_list.append(x1)
        if 2 in input_list:
            picked_list.append(x2)
        if 3 in input_list:
            picked_list.append(x12)
        if 4 in input_list:
            picked_list.append(x22)
        if 5 in input_list:
            picked_list.append(sinx1)
        if 6 in input_list:
            picked_list.append(sinx2)
        x = tf.transpose(tf.stack(picked_list))
        print("x is "+str(x.get_shape()))

    hidden1 = nn_layer(x, len(picked_list), hidden_dimension, 'layer1')
    #print("hidden1 is "+str(hidden1.get_shape()))

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(hidden1, keep_prob)
    #print("dropped is "+str(dropped.get_shape()))
    
    # Do not apply softmax activation yet, see below.
    y = nn_layer(dropped, hidden_dimension, 2, 'output', act=tf.identity)
    #print("y is "+str(y.get_shape()))

    print ("Creating summaries")
    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
        #                               reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the
        # raw outputs of the nn_layer above, and then average across
        # the batch.
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
                cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries and write them out to
    # /tmp/tensorflow/mnist/logs/mnist_with_summaries (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')
    print ("Initializing global variables")
    tf.global_variables_initializer().run()

    print("Adding graphing nodes")
    plot_buf_truth = tf.placeholder(tf.string)
    image_truth = tf.image.decode_png(plot_buf_truth, channels=4)
    image_truth = tf.expand_dims(image_truth, 0) # make it batched
    plot_image_summary_truth = tf.summary.image('truth', image_truth, max_outputs=1)
    
    plot_buf_train = tf.placeholder(tf.string)
    image_train = tf.image.decode_png(plot_buf_train, channels=4)
    image_train = tf.expand_dims(image_train, 0) # make it batched
    plot_image_summary_train = tf.summary.image('train', image_train, max_outputs=1)
    
    def feed_dict(train, _data, _labels):
        """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
        if train:
            xs, ys = _data[int(len(_data)/2):], _labels[int(len(_labels)/2):]
            k = FLAGS.dropout
        else:
            xs, ys = _data[:int(len(_data)/2)], _labels[:int(len(_labels)/2)]
            k = 1.
        #print("Size of data %d, size of labels %d" % (len(xs), len(ys)))
        #return {plot_buf_truth: get_plot_buf(xs, ys), xinput: xs, y_: ys, keep_prob: k}
        return {xinput: xs, y_: ys, keep_prob: k}


    plot_buf = get_plot_buf(input_data, input_labels)
    plot_image_summary_ = sess.run(
        plot_image_summary_truth,
        feed_dict={plot_buf_truth: plot_buf.getvalue()})
    train_writer.add_summary(plot_image_summary_, global_step=0)
    print("Starting to train")
    for i in range(FLAGS.max_steps):
#        print("Current training step is %d" % i)
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc, ce, y_eval, yeval = sess.run(
                [merged, accuracy, cross_entropy, y_, y],
                feed_dict=feed_dict(True, input_data, input_labels))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
            print('Cross-entropy at step %s: %s' % (i, ce))
            print('y_ at step %s: %s' % (i, str(y_eval[0:9].transpose())))
            print('y at step %s: %s' % (i, str(yeval[0:9].transpose())))
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _, xinputeval, y_eval, yeval = sess.run([merged, train_step, xinput, y_, y],
                                        feed_dict=feed_dict(False, input_data, input_labels),
                                        options=run_options,
                                        run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)

                plot_buf = get_plot_buf(xinputeval, yeval)
                plot_image_summary_ = sess.run(
                    plot_image_summary_train,
                    feed_dict={plot_buf_train: plot_buf.getvalue()})
                train_writer.add_summary(plot_image_summary_, global_step=i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(False, input_data, input_labels))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
    print("TRAINED.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimension', type=int, default=10,
        help='Number of samples to generate.')
    parser.add_argument('--dropout', type=float, default=0.9,
        help='Keep probability for training dropout.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
        help='Initial learning rate')
    parser.add_argument('--max_steps', type=int, default=1000,
        help='Number of steps to run trainer.')
    parser.add_argument('--noise', type=float, default=0,
        help='Amount of noise in [0,1] to use.')
    parser.add_argument('--data_type', type=int, default=SPIRAL,
        help='Which data set to use: two circles, squares, two clusters, spiral.')
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
             'tensorflow/Playground/logs/playground_example_with_summaries'),
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

