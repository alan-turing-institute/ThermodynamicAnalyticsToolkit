./LossFunctionSampler \
  --batch_data_files dataset-twoclusters.csv \
  --batch_size 20 \
  --csv_file LossFunctionSampler-output-SGLD.csv \
  --fix_parameter "output/biases/Variable:0=2." \
  --interval_weights -5 5 \
  --interval_biases -1 1 \
  --samples_weights 10 \
  --samples_biases 4
