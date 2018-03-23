# set up feed_dict
features, _ = model.input_pipeline.next_batch(model.sess)
feed_dict = { model.xinput: features }

# evaluate the output "y" nodes
y_node = model.nn.get_list_of_nodes(["y"])
y_eval = model.sess.run(y_node, feed_dict=feed_dict)
print(y_eval)
