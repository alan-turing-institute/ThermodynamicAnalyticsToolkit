DDSAnalyser \
    --diffusion_map_file diffusion_map_values.csv \
    --diffusion_map_method vanilla \
    --diffusion_matrix_file diffusion_map_vectors.csv \
    --drop_burnin 100 \
    --every_nth 10 \
    --inverse_temperatur 1e4 \
    --number_of_eigenvalues 4 \
    --steps 10 \
    --trajectory_file trajectory.csv