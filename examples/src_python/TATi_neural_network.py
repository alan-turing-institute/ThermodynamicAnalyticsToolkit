import numpy as np
import csv

from TATi.models.model import model

class NeuralNetwork():
    """
    This class uses allows to compute gradient and loss of a neural network using TATi.
    :param dataset: dictionary with ndarray X of data points under the key 'X', and ndarray Y of labels under the key 'Y'
    :param flags: dictionary, provide dictionary of parameters for creation of neural network in TATi. See TATi documentation. Default is None.
    :param optimize_for_initial_condition: bool, choose initial value of the parameters by performing 100 steps of gradient descent optimization (this can be changed in the flags dictionary).  If chosen False, the initial values are set to zero. default True
    """

    def __init__(self, XY, flags = None, optimize_for_initial_condition = True):

        X = XY['X']
        Y = XY['Y']
        self.batch_size = X.shape[0]
        self.write_data(X, Y, "dataset.csv")

        if flags is None:
            # prepare parameters
            self.FLAGS = model.setup_parameters(
                        hidden_activation="relu",
                        hidden_dimension="2 1",
                        output_activation="linear",
                        batch_data_files=["dataset.csv"],
                        batch_size=self.batch_size,
                        optimizer="GradientDescent",
                        max_steps=100,
                        step_width=1e-2
                    )
        else:
            self.FLAGS = flags

        self.nn = model(self.FLAGS)

        if optimize_for_initial_condition:
            print('Create model for initial optimization')
            self.nn.init_network(None, setup="train", add_vectorized_gradients=True)

            self.nn.init_input_pipeline()
            opt_run_info, opt_trajectory, _ = self.nn.train( return_run_info=True, return_trajectories=True)
            self.FLAGS.max_steps = 1000
            self.nn.reset_parameters(self.FLAGS)

            weights_eval = self.nn.weights.evaluate(self.nn.sess)
            biases_eval = self.nn.biases.evaluate(self.nn.sess)

            # assign intial value of the array of parameters
            self.nn_parameters =np.copy(np.concatenate((weights_eval, biases_eval)))


        print('Creating NN model for accessing forces')
        # prepare network and input pipeline
        self.nn.init_network(None, setup="None", add_vectorized_gradients=True)
        self.nn.init_input_pipeline()

        if optimize_for_initial_condition:
            self.nn.assign_neural_network_parameters(self.nn_parameters)
        else:

            # assign parameters of NN
            total_dof = self.nn.get_total_weight_dof() + self.nn.get_total_bias_dof()

            self.nn_parameters = np.zeros([total_dof])
            # ... assign parameters e.g. through parameter update directly
            # in the np array, then call ...
            self.nn.assign_neural_network_parameters(self.nn_parameters)

        # setup feed dict and evaluation nodes and evaluate loss
        features, labels = self.nn.input_pipeline.next_batch(self.nn.sess, auto_reset=True,  warn_when_reset = False)

        feed_dict = {
                    self.nn.xinput: features,
                    self.nn.nn.placeholder_nodes["y_"]: labels}
        # test gradient and loss
        loss, gradient_loss = self.nn.sess.run([self.nn.loss, self.nn.gradients],
                                               feed_dict=feed_dict)

        print("Initial neural net parameter values: \n "+repr(self.nn_parameters))

    def force(self,theta):
        """
        :param theta: ndarray in shape of self.nn_parameters
        :return : force is returned. Note that force is minus gradient of the loss, which is an ndarray in shape of self.nn_parameters.
        """
        self.nn.assign_neural_network_parameters(theta)

        features, labels = self.nn.input_pipeline.next_batch(self.nn.sess, auto_reset=True,  warn_when_reset = False)

        feed_dict = {
            self.nn.xinput: features,
            self.nn.nn.placeholder_nodes["y_"]: labels}

        self.loss_eval, self.gradient_eval = self.nn.sess.run([self.nn.loss, self.nn.gradients],feed_dict=feed_dict)

        # returns force, which is - gradient
        return -self.gradient_eval

    def loss(self,theta):
        """
        :param theta: ndarray in shape of self.nn_parameters
        :return : return loss
        """
        self.nn.assign_neural_network_parameters(theta)

        features, labels = self.nn.input_pipeline.next_batch(self.nn.sess, auto_reset=True,  warn_when_reset = False)

        feed_dict = {
            self.nn.xinput: features,
            self.nn.nn.placeholder_nodes["y_"]: labels}

        self.loss_eval = self.nn.sess.run(self.nn.loss, feed_dict=feed_dict)

        return self.loss_eval

    def predict(self, theta):

        self.nn.assign_neural_network_parameters(theta)
        # set up feed_dict
        features, _ = self.nn.input_pipeline.next_batch(self.nn.sess,  warn_when_reset = False)
        feed_dict = { self.nn.xinput: features }
        # evaluate the output "y" nodes
        y_node = self.nn.nn.get_list_of_nodes(["y"])
        y_eval = self.nn.sess.run(y_node, feed_dict=feed_dict)

        return y_eval[0]

    def accuracy(self, theta, print_score = True):

        self.nn.assign_neural_network_parameters(theta)

        features, labels = self.nn.input_pipeline.next_batch(self.nn.sess,  warn_when_reset = False)
        feed_dict = {
            self.nn.xinput: features,
            self.nn.nn.placeholder_nodes["y_"]: labels}

           # evaluate the "accuracy" node
        accuracy_node = self.nn.nn.get_list_of_nodes(["accuracy"])
        acc_eval = self.nn.sess.run(accuracy_node, feed_dict=feed_dict)[0]

        if print_score:
            print('Accuracy on training set is '+repr( acc_eval))

        return acc_eval

    def write_data(self, xs, ys, filename, shuffle = True):
        # always shuffle data set is good practice
        if shuffle:
            randomize = np.arange(len(xs))
            np.random.shuffle(randomize)
            xs[:] = np.array(xs)[randomize]
            ys[:] = np.array(ys)[randomize]

        with open(filename, 'w', newline='') as data_file:
            csv_writer = csv.writer(data_file, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            header = ["x"+str(i+1) for i in range(len(xs[0]))]+["label"]
            csv_writer.writerow(header)
            ys=ys[:,np.newaxis]

            for i in range(len(xs)):
                x = xs[i,:]
                y = ys[i,:]
                csv_writer.writerow(['{:{width}.{precision}e}'.format(val, width=8,precision=8) for val in list(x)] + ['{}'.format(y[0])])
        data_file.close()
