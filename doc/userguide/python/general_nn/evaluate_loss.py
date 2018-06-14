# set up feed_dict
features, labels = model.input_pipeline.next_batch(model.sess)
feed_dict = {
    model.xinput: features,
    model.nn.placeholder_nodes["y_"]: labels}

# evaluate the "loss" node
loss_eval = model.sess.run(model.loss, feed_dict=feed_dict)
print(loss_eval)
