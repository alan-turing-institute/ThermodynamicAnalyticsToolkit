TATiOptimizer \
    --batch_data_files dataset-twoclusters.csv \
    --batch_size 50 \
    --loss mean_squared \
	--learning_rate 1e-2 \
    --max_steps 1000 \
    --optimizer GradientDescent \
    --run_file run.csv \
    --save_model `pwd`/model.ckpt.meta \
    --seed 426
