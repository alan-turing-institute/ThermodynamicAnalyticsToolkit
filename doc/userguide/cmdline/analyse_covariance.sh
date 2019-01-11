TATiAnalyser \
    covariance \
    --covariance_matrix covariance.csv \
    --covariance_eigenvalues eigenvalues.csv \
    --covariance_eigenvectors eigenvectors.csv \
    --drop_burnin 100 \
    --every_nth 10 \
    --number_of_eigenvalues 3 \
    --trajectory_file trajectory.csv