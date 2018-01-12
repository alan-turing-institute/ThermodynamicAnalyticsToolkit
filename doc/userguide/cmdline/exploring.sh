DDSExplorer \
  	--batch_data_files dataset-twoclusters.csv \
    --batch_size 50 \
    --diffusion_map_method vanilla \
    --friction_constant 10 \
    --inverse_temperature 10 \
    --loss mean_squared \
    --max_legs 10 \
    --max_steps 10 \
    --number_of_eigenvalues 1 \
    --number_of_parallel_trajectories 1 \
    --sampler GeometricLangevinAlgorithm_2ndOrder \
    --run_file run.csv \
    --seed 426 \
    --step_width 1e-2 \
    --trajectory_file trajectory.csv \
    --use_reweighting 0
