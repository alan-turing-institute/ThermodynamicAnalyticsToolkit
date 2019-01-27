import TATi.simulation as tati

nn = tati(
    verbose=1,
)
print(nn.get_options())
nn.set_options(verbose=2)
print(nn.get_options("verbose"))
