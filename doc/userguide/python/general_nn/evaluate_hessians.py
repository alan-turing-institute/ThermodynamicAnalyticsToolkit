# requires "do_hessians=True" in FLAGS before init_network()

# setup feed dict and evaluation nodes and evaluate loss
features, labels = model.input_pipeline.next_batch(model.sess)
feed_dict = {
    model.xinput: features,
    model.nn.placeholder_nodes["y_"]: labels}

# evaluate the gradients
hessian_eval = model.sess.run([model.hessians],
                              feed_dict=feed_dict)
print(hessian_eval)
