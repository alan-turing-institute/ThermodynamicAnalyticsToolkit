TATiLossFunctionSampler \
  naive_grid \
  --batch_data_files dataset-twoclusters.csv \
  --batch_size 20 \
  --csv_file TATiLossFunctionSampler-output-SGLD.csv \
  --fix_parameters "output/weights/Variable:0=2.,2." \
  --interval_weights -5 5 \
  --interval_biases -1 1 \
  --samples_weights 10 \
  --samples_biases 4
