TATiLossFunctionSampler \
  naive_grid \
  --batch_data_files dataset-twoclusters.csv \
  --batch_size 20 \
  --csv_file TATiLossFunctionSampler-output-SGLD.csv \
  --exclude_parameters "w0" \
  --fix_parameters "output/biases/Variable:0=0" \
  --interval_weights -5 5 \
  --interval_biases -1 1 \
  --parse_parameters_file centers.csv \
  --parse_steps 1 \
  --samples_weights 10 \
  --samples_biases 4 \
  -vv
