import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    seed=427,
)
print("There are "+str(nn.num_parameters())+" momenta.")
print("Momenta: "+str(nn.momenta))
