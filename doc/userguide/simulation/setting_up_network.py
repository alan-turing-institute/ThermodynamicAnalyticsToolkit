import TATi.simulation as tati

# prepare parameters
nn = tati(
	batch_data_files=["dataset-twoclusters.csv"],
	hidden_dimension=[8, 8],
	hidden_activation="relu",
	output_activation="linear",
	seed=427,
)
print(nn.num_parameters())
