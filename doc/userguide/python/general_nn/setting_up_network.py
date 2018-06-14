from TATi.models.model import tati_model

import numpy as np

# prepare parameters
FLAGS = tati_model.setup_parameters(
	hidden_dimension="8 8",
	hidden_activation="relu",
	output_activation="linear",
)
model = tati_model(FLAGS)

# prepare network
model.init_network(None, setup="None")
