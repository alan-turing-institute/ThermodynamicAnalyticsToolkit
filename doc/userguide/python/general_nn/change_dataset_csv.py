# change the CSV in the FLAGS and re-create the input pipeline
FLAGS.batch_data_files = ["different_dataset.csv"]
model.create_input_pipeline(parameters)
