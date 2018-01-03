from DataDrivenSampler.datasets.classificationdatasets \
    import ClassificationDatasets as DatasetGenerator

import csv
import numpy as np

np.random.seed(426)

dataset_generator = DatasetGenerator()
ds = dataset_generator.generate(
    dimension=100,
    noise=0.1,
    data_type=dataset_generator.TWOCLUSTERS)

with open("dataset-twoclusters.csv", 'w', newline='') as data_file:
    csv_writer = csv.writer(data_file, delimiter=',', \
                            quotechar='"', \
                            quoting=csv.QUOTE_MINIMAL)
    header = ["x"+str(i+1) for i in range(len(ds.xs[0]))]+["label"]
    csv_writer.writerow(header)
    for x, y in zip(ds.xs, ds.ys):
        csv_writer.writerow(list(x)+[y[0]])
    data_file.close()
