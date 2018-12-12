import TATi.simulation as tati

# prepare parameters
nn = tati(
	batch_data_files=["dataset-twoclusters.csv"],
	hidden_dimension=[8, 8],
	hidden_activation="relu",
	input_dimension=2,
	loss="mean_squared",
	output_activation="linear",
	output_dimension=1,
	seed=427,
)
print(nn.num_parameters())
