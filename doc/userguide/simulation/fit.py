import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    learning_rate=0.03,
    max_steps=100,
    optimizer="GradientDescent",
    seed=427,
)
print("Start: "+str(nn.loss()))
nn.fit()
print("End  : "+str(nn.loss()))
