import TATi.simulation as tati

# prepare parameters
nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    seed=427,
)
print(nn.dataset)