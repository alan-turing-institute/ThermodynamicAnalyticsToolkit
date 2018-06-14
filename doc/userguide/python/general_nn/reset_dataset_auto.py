# set the input pipeline for auto reset
features, labels = model.input_pipeline.next_batch(model.sess,
                                                   auto_reset=True)

