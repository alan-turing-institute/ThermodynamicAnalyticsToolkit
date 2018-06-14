# set up feed_dict
features, labels = model.input_pipeline.next_batch(model.sess)
feed_dict = {
    model.xinput: features,
    model.nn.placeholder_nodes["y_"]: labels}

# evaluate the "accuracy" node
accuracy_node = model.nn.get_list_of_nodes(["accuracy"])
acc_eval = model.sess.run(accuracy_node, feed_dict=feed_dict)
print(acc_eval)
