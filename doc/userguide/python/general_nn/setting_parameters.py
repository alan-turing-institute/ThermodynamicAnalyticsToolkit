# assign parameters of NN
total_dof = model.get_total_weight_dof()+model.get_total_bias_dof()
nn_parameters = np.zeros([total_dof])
# ... assign parameters e.g. through parameter update directly
# in the np array, then call ...
model.assign_neural_network_parameters(nn_parameters)
