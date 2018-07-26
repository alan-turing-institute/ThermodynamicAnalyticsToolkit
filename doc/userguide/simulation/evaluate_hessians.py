import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    do_hessians=True,
    seed=427
)
print(nn.hessians())
