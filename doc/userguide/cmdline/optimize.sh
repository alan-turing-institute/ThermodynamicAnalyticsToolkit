DDSOptimizer \
    --batch_data_files dataset-twoclusters.csv \
    --batch_size 50 \
    --loss mean_squared \
    --max_steps 1000 \
    --optimizer GradientDescent \
    --run_file run.csv \
    --save_model `pwd`/model.ckpt.meta \
    --seed 426 \
    --step_width 1e-2
