import TATi.simulation as tati

nn = tati(batch_size=2)
nn.dataset = ["dataset-twoclusters.csv"]
print(nn.dataset)