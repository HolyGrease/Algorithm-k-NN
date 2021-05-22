import math

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

def minkowsky(x, y, w):
	return 0

def manhattan(x, y, w):
	return 0

def camberra(x, y, w):
	return 0

def chebychev(x, y, w):
	return 0

def quadratic(x, y, w, q):
	return 0

def mahalanobis(x, y, w, v):
	return 0

def correlation(x, y, w):
	return 0

def chi_square(x, y, w, sums):
	return 0

def kendalls(x, y, w):
	return 0