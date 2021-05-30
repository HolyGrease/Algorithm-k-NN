import math

from itertools import groupby

from dataset import Dataset

def k_NN(dataset, x, k, metric, attributes_weights=None, distance_weight=None):
	"""Method clasify row

	Args:
		dataset: 					dataset which used to classify
		x (list): 					row to be classified
		k (int): 					number of neighbors, must be large then 0
		metric (predicate): 		metric by which calculate distance
		attributes_weights (list): 	weights for attributes
		distance_weight (list): 	weights for distances

	Returns:
		object: 					class. Type depends on target value type
		None: 						if k is incorrect
	"""
	# Check k
	if (k < 1):
		return None
	# If attributes weight not specify
	if attributes_weights is None:
		# Fill by 1, don't count target attribute
		attributes_weights = [1 for i in range(dataset.get_columns_number() - 1)]
	# If attributes_weights specify - check for same dimension
	# don't count target attribute
	elif len(attributes_weights) != len(dataset.get_columns_number() - 1):
		return None
	# If distance weight not specify
	if distance_weight is None:
		# Fill by 1
		distance_weight = [1 for i in range(k)]
	# If distance weight specify - check number of weights to equal to k
	elif len(distance_weight) != k:
		return None
	# Count distance between two dots
	output = []
	for index, row in enumerate(dataset.data):
		# Make copy of the row
		value = row.copy()
		# Delete target element
		value.pop(dataset.target)
		# Calculate distance
		output.append((distance(x, value, attributes_weights, metric), index))
	# Sort by distance descending
	output.sort()
	# Get first k elements
	neighbours = output[:k]
	# Get only indexes
	classes = map(lambda x: dataset.data[x[1]][dataset.target], neighbours)
	# Count number of instances in each class
	frequency = [(len([i for i in value]), key) for key, value in groupby(sorted(classes), None)]
	# Voting
	voting = [frequency[i] * distance_weight[i] for i in range(len(frequency))]
	# Get class
	cl = sorted(voting)[-1]
	# Return class
	return cl[1]

def distance(x, y, weight, metric):
	"""Function calculate distance between two dots
		using specific metric and weights for each dimension

	Args:
		x (list): first dot
		y (list): second dot
		weight (list): weights determine how specific dimension
			influence on general distance
		metric (predicate): metric that used to calculate distance
	"""
	# If weight not specified
	if weight is None:
		weight = [1 for i in range(len(x))]
	# Check coordinates for same length (dimension)
	if len(x) != len(y) or len(x) != len(weight):
		return -1
	# Calculate distance
	return metric(x, y, weight)

def euclidean(x, y, w):
	"""Function calculate Euclidean distance

	Args:
		x (list): first dot
		y (list): second dot
		w (list): weights determine how specific dimension
	"""
	return math.sqrt(sum(
		math.pow(x[i] - y[i], 2) * w[i]
		for i in range(len(x))
	))

# TODO implement other metrics

def manhattan(x, y, w):
	"""Function calculate Manhattan distance

	Args:
		x (list): first dot
		y (list): second dot
		w (list): weights determine how specific dimension
	"""
	return sum(
		math.fabs(x[i] - y[i]) * w[i]
		for i in range(lex(x)))

def camberra(x, y, w):
	"""Function calculate Camberra distance

	Args:
		x (list): first dot
		y (list): second dot
		w (list): weights determine how specific dimension
	"""
	return sum(
		math.fabs((x[i] - y[i]) / (x[i] + y[i])) * w[i]
		for i in range(lex(x)))

def chebychev(x, y, w):
	"""Function calculate Chebychev distance

	Args:
		x (list): first dot
		y (list): second dot
		w (list): weights determine how specific dimension
	"""
	return max(
		math.fabs(x[i] - y[i]) * w[i]
		for i in range(lex(x)))