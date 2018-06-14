# e.g. parse dataset from CSV file into pandas frame
input_dimension = 2
output_dimension = 1
parsed_csv = pd.read_csv("dataset.csv", sep=',', header=0)
# extract feature and label columns as numpy arrays
features = np.asarray(\
	parsed_csv.iloc[:, 0:input_dimension])
labels = np.asarray(\
	parsed_csv.iloc[:, \
	input_dimension:input_dimension \
	+ output_dimension])

# supply dataset (this creates the input layer)
model.provide_data(features, labels)

