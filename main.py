import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from distances import*
from data import Dataset

# Google doc style https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

def iris_dataset():
	filename = "resources\\data\\iris\\iris.data"
	csv_reader = csv.reader(open(filename), delimiter=",")
	data = []
	for row in csv_reader:
		temp = []
		for i in range(4):
			temp.append(float(row[i]))
		temp.append(row[4])
		data.append(temp)


	names = ["sl", "sw", "pl", "pw", "class"]
	dataset = Dataset(data, 4, names, "iris")

	return dataset

def wine_dataset():
	filename = "resources\\data\\wine\\wine.data"
	csv_reader = csv.reader(open(filename), delimiter=",")
	data = []
	for row in csv_reader:
		temp = []
		for i in range(1, 13):
			temp.append(float(row[i]))
		temp.append(row[0])
		data.append(temp)


	names = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium",
		"Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins",
		"Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
	dataset = Dataset(data, 12, names, "wine")

	return dataset

def concrete_dataset():
	filename = "resources\\data\\concrete\\Concrete_Data.data"
	csv_reader = csv.reader(open(filename), delimiter=",")
	data = []
	for row in csv_reader:
		temp = []
		for i in range(9):
			temp.append(float(row[i]))
		data.append(temp)

	names = ["Cement", "Slag", "Ash", "Water", "Superplasticizer", "Coarse Aggregate",
	"Fine Aggregate", "Strength", "Age"]
	dataset = Dataset(data, 8, names, "concrete")

	return dataset

def main():
	# Get datasets
	iris = iris_dataset()
	wine = wine_dataset()
	concrete = concrete_dataset()
	# Prepocess datasets
	for i in range(4):
		iris.normalize(i)

	for i in range(12):
		wine.normalize(i)

	for i in range(8):
		wine.normalize(i)

	# !For debug!

	# Save results
	results = []
	# Test algorithms on datasets
	results.append(all_tests(iris))
	print()
	results.append(all_tests(wine))
	print()
	results.append(all_tests(concrete))

	results_3_NN = {}
	names = ["iris", "wine", "concrete"]
	for i in range(len(results)):
		results_3_NN[names[i]] = results[i][3]

	plot_by_results(results_3_NN, "3-NN")
	plt.show()

def all_tests(dataset):
	if dataset == None:
		return None
	
	# Test without weights
	r = test(dataset, name="simple")
	print()
	
	# Test with weights for distance
	distance_weights = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
	test(dataset, None, distance_weights, "distance_weights")
	print()

	# Test with weights for attributes
	# Get index of target column
	target = dataset.index_target
	# Count weight for attributes using information gain
	attribute_weights = [
		dataset.gain(i)
		for i in range(dataset.get_column_number())
		if i != target]
	test(dataset, attribute_weights, name="attribute_weights")
	print()

	# Test with both weights
	test(dataset, attribute_weights, distance_weights, name="combine")

	return r

def test(dataset, attribute_weights=None, distance_weights=None, name=None):
	# Shuffle dataset
	shuffled = dataset.shuffle()
	# Dictionary to write results of tests
	results = {}
	# Split dataset for train and test
	for ratio in np.arange(0.1, 1, 0.1):
		# Split dataset to train and test by specific ratio
		train, test = shuffled.split_by_ratio(ratio)
		# Test k-NN methods with k from 1 to 9
		for k in range(1, 8, 2):
			# Predict class for each row in test data
			classes = [train.k_NN(k, row, attribute_weights, distance_weights, euclidean) for row in test.get_data()]
			# Count accuracy of prediction
			accuracy = count_accuracy(test.get_target_column(), classes)
			# Add accuraacy to dictionary where k is key
			# If k not in dictionary
			if not k in results:
				# Add pair (ratio, accuracy), where ratio - ration size of train dataset
				results[k] = [(ratio, accuracy)]
			# Otherwise
			else:
				# Add pair (ratio, accuracy), where ratio - ration size of train dataset
				results[k].append((ratio, accuracy))
	save_as_csv(results, "out\\" + dataset.name + "\\" + name)
	# Print results as table
	print_as_table(results)
	# Create and show plot by results
	plot_by_results(results, dataset.name + " " + name)
	# Return results
	return results

def save_as_csv(results, name):
	file = open(name + ".csv", "w")

	first_column_name = "Method name"
	# Firs column contains methods names
	file.write(first_column_name + ";")
	for values in results[list(results.keys())[0]]:
		# Get ratio
		ratio = values[0]
		file.write("{:1.3f}".format(ratio) + ";")
	# New line
	file.write("\n")

	# Print accuraces in table format
	for key, values in results.items():
		# Print method name
		file.write(f"{key}-NN method" + ";")
		# For each value of this method values
		for value in values:
			# Get accuracy
			accuracy = value[1]
			# Print accuracy in specific format
			file.write("{:3.3f}".format(accuracy) + ";")
		# New line
		file.write("\n")

def print_as_table(results):
	"""Function print to terminal results of test in table format

	Args:
		results (Dictionary): Represent test result in specific format
			{"Method1": [(ratio1, accuracy11), (ratio2, accuracy12)...]
			 "Method2"}
	"""
	# Print table header
	first_column_name = "Method name"
	# Firs column contains methods names
	print(first_column_name, end=" | ")
	for values in results[list(results.keys())[0]]:
		# Get ratio
		ratio = values[0]
		print("{:1.3f}".format(ratio), end=" | ")
	# New line
	print()

	# Print delimetr between table header and body
	for i in range(len(first_column_name) + 1):
		print("-", end="")
	print("+-", end="")
	for i in range(len(results[list(results.keys())[0]]) - 1):
		print("------", end="+-")
	# Last column ends only with "+" and go to new line
	print("------", end="+\n")

	# Print accuraces in table format
	for key, values in results.items():
		# Print method name
		print(f"{key}-NN method", end=" | ")
		# For each value of this method values
		for value in values:
			# Get accuracy
			accuracy = value[1]
			# Print accuracy in specific format
			print("{:3.3f}".format(accuracy), end=" | ")
		# New line
		print()

def plot_by_results(results, name=None):
	"""Function create and show plot
	
	Args:
		results (Dictionary): Represent test result in specific format
			{"Method1": [(ratio1, accuracy11), (ratio2, accuracy12)...]
			 "Method2"}
	"""
	_, new_plot = plt.subplots(1)
	lines_style = ["-", "--", "-.", ":"]
	# For each method in dictionary create it's own  graph
	for key, values in results.items():
		# Split coordinates
		x = [v[0] for v in values]
		y = [v[1] for v in values]

		f = interp1d(x, y, kind="cubic")

		xnew = np.linspace(min(x), max(x), num=100, endpoint=True)

		# As x axis takes ration, as y - accuracy
		# also add graph name same as method name
		new_plot.plot(
			xnew,
			f(xnew),
			lines_style.pop(0),
			label=f"{key}-NN")
	# Add legend to the plot
	new_plot.legend()
	# Set axis labels
	new_plot.set_xlabel("Ration (of train dataset)")
	new_plot.set_ylabel("Acurracy")
	# Set plot title
	if name != None:
		new_plot.set_title(name)
	# Show plot

def count_accuracy(asserted_values, values):
	"""Function count accuracy

	Args:
		asserted_values (List): List of expected values
		values (List): List of actual values

	Returns:
		float: Ration of correct values to all values
			or -1 if arguments have different length
	"""
	# If arguments have diffenrent length
	if len(asserted_values) != len(values):
		# Return -1 as fail
		return -1
	# Define correct values counter
	correct = 0
	# Define wrong values counter
	wrong = 0
	# Compare values in lists by same index
	for i in range(len(asserted_values)):
		# If values equal
		if asserted_values[i] == values[i]:
			# Increment correct values counter
			correct += 1
		# Othewise
		else:
			# Increment wrong values counter
			wrong += 1
	# Return ration of correct values to all values
	return correct / (correct + wrong)

if __name__ == '__main__':
	main()