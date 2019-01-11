from TATi.datasets.classificationdatasets \
    import ClassificationDatasets as DatasetGenerator
from TATi.common import data_numpy_to_csv

import numpy as np

# fix random seed for reproducibility
np.random.seed(426)

# generate test dataset: two clusters
dataset_generator = DatasetGenerator()
xs, ys = dataset_generator.generate(
    dimension=500,
    noise=0.01,
    data_type=dataset_generator.TWOCLUSTERS)

# always shuffle data set is good practice
randomize = np.arange(len(xs))
np.random.shuffle(randomize)
xs[:] = np.array(xs)[randomize]
ys[:] = np.array(ys)[randomize]

# call helper to write as properly formatted CSV
data_numpy_to_csv(xs,ys, "dataset-twoclusters.csv")
