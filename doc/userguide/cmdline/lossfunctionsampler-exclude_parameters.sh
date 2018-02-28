LossFunctionSampler \
  --batch_data_files dataset-twoclusters.csv \
  --batch_size 20 \
  --csv_file LossFunctionSampler-output-SGLD.csv \
  --exclude_parameters "w1" \
  --interval_weights -5 5 \
  --interval_biases -1 1 \
  --samples_weights 10 \
  --samples_biases 4
