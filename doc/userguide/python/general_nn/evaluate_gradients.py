# setup feed dict and evaluation nodes and evaluate loss
features, labels = model.input_pipeline.next_batch(model.sess)
feed_dict = {
    model.xinput: features,
    model.nn.placeholder_nodes["y_"]: labels}

# evaluate the gradients
gradient_eval = model.sess.run([model.gradients],
                               feed_dict=feed_dict)
print(gradient_eval)
