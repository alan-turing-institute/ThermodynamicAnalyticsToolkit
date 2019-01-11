from TATi.model import Model as tati

import numpy as np

# prepare parameters
FLAGS = tati.setup_parameters(
    batch_data_files=["dataset-twoclusters.csv"],
    batch_size=10,
    output_activation="linear"
)
nn = tati(FLAGS)

# prepare network and input pipeline
nn.init_input_pipeline()
nn.init_network(None, setup="None",
                add_vectorized_gradients=True)
nn.reset_dataset()

# assign parameters of NN
total_dof = nn.get_total_weight_dof()+nn.get_total_bias_dof()
nn_parameters = np.zeros([total_dof])
# ... assign parameters e.g. through parameter update directly
# in the np array, then call ...
nn.assign_neural_network_parameters(nn_parameters)

# setup feed dict and evaluation nodes and evaluate loss
features, labels = nn.input_pipeline.next_batch(nn.sess,
                                                auto_reset=True)
feed_dict = {
    nn.xinput: features,
    nn.nn[0].placeholder_nodes["y_"]: labels}

# simply evaluate loss
loss_eval = nn.sess.run(nn.loss, feed_dict=feed_dict)
print(loss_eval)

# alternativly, evaluate loss and gradients. otherwise loss and 
# gradient may not match due to different subset of dataset
# (if batch_size != dimension)
loss_eval, gradient_eval = nn.sess.run([nn.loss, nn.gradients],
                                       feed_dict=feed_dict)
print(gradient_eval)

