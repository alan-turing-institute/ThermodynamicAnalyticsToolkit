from TATi.common import data_numpy_to_csv
import numpy as np

X = np.asarray([[1]])
Y = np.asarray([[0]])

# prepare and save the trivial data set for later
datasetName = 'dataset_ho.csv'
data_numpy_to_csv(X, Y, datasetName)
numberOfPoints = 1
