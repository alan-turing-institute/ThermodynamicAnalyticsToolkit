import TATi.simulation as tati

nn = tati(
    batch_data_files=["dataset-twoclusters.csv"],
    friction_constant=1e1,
    inverse_temperature=1e3,
    max_steps=100,
    sampler="BAOAB",
    seed=427,
    step_width=0.1,
)
print("Start: "+str(nn.loss()))
nn.sample()
print("End  : "+str(nn.loss()))
