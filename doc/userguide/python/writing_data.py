from TATi.datasets.classificationdatasets \
    import ClassificationDatasets as DatasetGenerator

import csv
import numpy as np

np.random.seed(426)

dataset_generator = DatasetGenerator()
xs, ys = dataset_generator.generate(
    dimension=100,
    noise=0.1,
    data_type=dataset_generator.TWOCLUSTERS)

# always shuffle data set is good practice
randomize = np.arange(len(xs))
np.random.shuffle(randomize)
xs[:] = np.array(xs)[randomize]
ys[:] = np.array(ys)[randomize]

with open("dataset-twoclusters.csv", 'w', newline='') as data_file:
    csv_writer = csv.writer(data_file, delimiter=',', \
                            quotechar='"', \
                            quoting=csv.QUOTE_MINIMAL)
    header = ["x"+str(i+1) for i in range(len(xs[0]))]+["label"]
    csv_writer.writerow(header)
    for x, y in zip(xs, ys):
        csv_writer.writerow(
            ['{:{width}.{precision}e}'.format(val, width=8,
                                              precision=8)
             for val in list(x)] \
            + ['{}'.format(y[0], width=8,
                                                precision=8)])
    data_file.close()
