import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    fix_parameters="output/biases/Variable:0=2.",
    hidden_dimension=[8, 8],
    hidden_activation="relu",
    input_dimension=2,
    loss="mean_squared",
    output_activation="linear",
    output_dimension=1,
    seed=427,
)
print(nn.num_parameters())
