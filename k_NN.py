import math

from itertools import groupby

from dataset import Dataset

# TODO some check of arguments
# TODO check distance_weight for None, and fill it by 1 in this case
# then delete this check from distance
def k_NN(dataset, k, x, metric, attributes_weight=None, distance_weight=None):
	"""Method clasify row

	Args:
		dataset: 				dataset which used to classify
		k (int): 				number of neighbors, must be large then 0
		x (list): 				row to be classified
		metric (predicate): 	metric by which calculate distance
		attributes_weight (list): 	weights for attributes
		distance_weight (list): 		weights for distances

	Returns:
		object: 				class. Type depends on target value type
		None: 					if no neighbors or k is incorrect
	"""
	if (k < 1):
			return None
	# Count distance between two dots
	output = []
	for index, row in enumerate(dataset.data):
		# Make copy of the row
		value = row.copy()
		# Delete target element
		value.pop(dataset.target)
		# Calculate distance
		output.append((distance(x, value, distance_weight, metric), index))
	# Sort by distance descending
	output.sort()
	# Get first k elements
	neighbours = output[:k]
	# Get only indexes
	classes = map(lambda x: dataset.data[x[1]][dataset.target], neighbours)
	# Count number of instances in each class
	frequency = [(len([i for i in value]), key) for key, value in groupby(sorted(classes), None)]
	# If attributes coefficients are None
	if attributes_weight == None:
		attributes_weight = [1 for i in range(len(frequency))]
	# Check for same length (dimension)
	if len(attributes_weight) != len(frequency):
		return None
	# Voting
	voting = [frequency[i] * attributes_weight[i] for i in range(len(frequency))]
	# Get class
	cl = sorted(voting)[-1]
	# Return class
	return cl[1]

def distance(x, y, weight, metric):
	# If weight not specified
	if weight == None:
		weight = [1 for i in range(len(x))]

	# Check coordinates for same length (dimension)
	if len(x) != len(y) or len(x) != len(weight):
		return -1

	return metric(x, y, weight)

def euclidean(x, y, w):
	return math.sqrt(sum(
		math.pow(x[i] - y[i], 2) 
		for i in range(len(x))
	))

# TODO implement other metrics

def minkowsky(x, y, w):
	return 0

def manhattan(x, y, w):
	return 0

def camberra(x, y, w):
	return 0

def chebychev(x, y, w):
	return 0

def correlation(x, y, w):
	return 0

def kendalls(x, y, w):
	return 0