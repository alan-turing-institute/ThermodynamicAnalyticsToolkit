# modify parameters to contain dataset information
FLAGS.batch_data_files = ["dataset.csv"]
FLAGS.batch_data_file_type = ["csv"]
model.reset_parameters(FLAGS)

# create input and output layers, and nodes for reading file
model.init_input_pipeline()
