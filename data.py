import random

from itertools import groupby

from distances import*

class Dataset:
	def __init__(self, data, index_target, names, name):
		self.data = data
		self.index_target = index_target
		self.names = names
		self.name = name

	def print(self):
		print("target ", end="= ")
		print(self.index_target)
		for row in self.data:
			print(row)

	def get_data(self):
		return self.data

	def get_column(self, index):
		return [row[index] for row in self.data]

	def get_target_column(self):
		return self.get_column(self.index_target)

	def get_column_number(self):
		return len(self.data[0])

	def normalize(self, column_index):
		minimum = min(self.get_column(column_index))
		maximum = max(self.get_column(column_index))

		for row in self.data:
			row[column_index] = (row[column_index] - minimum) / (maximum - minimum)


	def entropy(self, column):
		"""Method calculate column entropy
		
		Args:
			column (List): list of values

		Returns:
			float: entropy of column. In the range [0, 1]
			None: if column is None
		"""
		# If column is None
		if column == None:
			# Return None
			return None
		# Group column by identity
		grouped = groupby(sorted(column))
		# Count rows number
		rows_number = len(column)
		# Count number of clases
		n = len(set(column))
		# If have only one class, in this case entropy equals 0
		if n == 1:
			# Return 0
			return 0
		# Calculate entropy
		output = 0
		# For each group
		for _, value in grouped:
			# Get list of values
			v = list(value)
			# Calculate
			output -= len(v) / rows_number * math.log(len(v)/rows_number, n)
		# Return entropy
		return output

	def gain(self, index, predicate=None):
		"""Method calculates gain of specific column

		Args:
			index (int): index of column
			predicate (function(x, y)): function that takes to arguments,
				used to group column

		Return:
			float: gain of column
			None: if index is wrong
		"""
		# Get column
		column = self.get_column(index)
		# If index is wrong
		if column == None:
			return None
		# Get target column
		target = self.get_target_column()
		# If dataset doesn't have column by this index
		# or target column index wrong
		if column == None or target == None:
			# Return None
			return None
		# Concat column and target column
		pairs = zip(column, target)
		# If predicate doesn't specified
		if predicate == None:
			unpacking_predicate = lambda x: x[0]
		# Otherwise
		else:
			# Unpacking predicate
			unpacking_predicate = lambda x: predicate(*x)
		# Group by predicate
		grouped = groupby(sorted(pairs, key=unpacking_predicate), unpacking_predicate)
		# Count row number
		rows_number = len(column)
		# Count entropy of target column
		output = self.entropy(self.get_target_column())
		# For each group
		for _, value in grouped:
			# Get list of values
			v = list(value)
			# Calculate
			output -= len(v) / rows_number * self.entropy(v)
		# Return entropy
		return output

	def k_NN(self, k, x, w_attributes, w_distance, metric):
		"""Method clasify instance

		Args:
			k (int): number if neighbors
			x (list):  row to be classified
			w_attributes (list) : weights for attributes
			w_distance (list) : weights fo distances
			metric (predicate) : type of metric 

		Returns:
			object: class. Type depends on target value type
			None: if k is illegal
		"""
		if (k < 1):
			return None
		# Delete target element
		x = x.copy()
		x.pop(self.index_target)
		# Count distance between two dots
		distances = []
		for index, row in enumerate(self.data):
			# Delete target element
			value = row.copy()
			value.pop(self.index_target)
			# Memorize the distance
			distances.append((distance(x, value, w_attributes, metric), index))
		# Sort by distance
		distances.sort()
		# Get first k elements
		neighbours = distances[:k]
		# Get only indexes
		classes = map(lambda x: self.data[x[1]][self.index_target], neighbours)
		# Voting for class
		voting = {}
		# If distance weight doesn't specified
		if w_distance == None:
			w_distance = [1 for i in range(k)]
		# Voting
		for index, value in enumerate(classes):
			if value in voting:
				voting[value] += w_distance[index]
			else:
				voting[value] = w_distance[index]
		# Get turple with most votes
		most_popular = max((value, key) for key, value in voting.items())
		# Get class
		cl = most_popular[1]
		# Return class
		return cl

	def split_by_ratio(self, ratio):
		TRAIN_DATASET_LEN = round(len(self.data) * ratio)

		train = []
		test = []

		for i in range(TRAIN_DATASET_LEN):
			train.append(self.data[i].copy())

		for i in range(TRAIN_DATASET_LEN, len(self.data)):
			test.append(self.data[i].copy())

		return (Dataset(train, self.index_target, self.names, self.name), Dataset(test, self.index_target, self.names, self.name))

	def shuffle(self):
		data = [row for row in self.data]

		random.shuffle(data)

		return Dataset(data, self.index_target, self.names, self.name)