import random
import csv
import math

from itertools import groupby


class Dataset():
	def __init__(self, data, target_index, columns_names, name):
		"""Method create Dataset item

		Args:
			data (list): 			list of table rows
			target_index (int): 	index of column with target attribute
			columns_names (list): 	list with columns names
		"""
		# Check data
		if data is None:
			raise ValueError("Data in None!")
		self._data = data
		# Check target_index for valid index
		if target_index >= self.get_columns_number():
			raise ValueError(f"Target index out of range! Got {target_index}, columns number: {self.get_columns_number()}")
		self._target_index = target_index
		# Check is number of columns names same as columns in data
		# If not raise exception
		if len(columns_names) != self.get_columns_number():
			raise ValueError(f"Wrong columns names number! Got {len(columns_names)}, columns number: {self.get_columns_number()}")
		self._columns_names = columns_names
		self._name = name

	@staticmethod
	def get_iris(path=None):
		"""Method return iris dataset
			by default dataset must be available
			by path "resources\\data\\iris\\iris.data"
			link to dataset "http://archive.ics.uci.edu/ml/datasets/Iris/"

		Args:
			path (string): Path to file with data

		Returns:
			Dataset: Dataset object based on iris data set
		"""
		# If path not specify
		if path is None:
			path = "resources\\data\\iris\\iris.data"

		# Open file as csv
		csv_reader = csv.reader(open(path), delimiter=",")
		# Save data to list
		data = []
		# For each row in file
		for row in csv_reader:

			temp = []
			# Read first 4 attributes as float
			for i in range(4):
				temp.append(float(row[i]))
			# Read last attribute as string
			temp.append(row[4])
			data.append(temp)

		# Set attributes names
		names = [
			"Sepal length", "Sepal width",
			"Petal length", "Petal width",
			"Class"]
		# Create Dataset
		dataset = Dataset(data, 4, names, "Iris")
		# return Dataset
		return dataset

	def print(self, rows_number=None):
		"""Method prints the dataset in console

		Args:
			rows_number (int): number of rows to print
		"""
		# If number of rows to print not specify
		if rows_number is None:
			# Set number of rows in dataset
			rows_number = self.get_rows_number()
		# Print dataset name
		print(f"Name = {self._name}")
		# Print name of target attributes
		print(f"Target = {self._columns_names[self._target_index]}")
		# Print column columns_names
		for name in self._columns_names:
			# Center align of text
			print("{:^15}".format(name), end=" | ")
		# Go to new line
		print()
		# Print data as table
		for i in range(rows_number):
			for attribute in self.data[i]:
				# Center align of text
				print("{:^15}".format(attribute), end=" | ")
			# Go to new line
			print()

	@property
	def data(self):
		return [row.copy() for row in self._data]

	@property
	def target(self):
		return self._target_index

	@target.setter
	def target(self, new_index):
		if new_index < get_columns_number and new_index >= 0:
			self._target_index = new_index

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, new_name):
		self._name = new_name

	def get_row(self, index):
		"""Method return row by index

		Args:
			index (int): index of row.
				Must be large than 0 and less than rows number in dataset

		Returns:
			list: row
		"""
		# Check is index valid
		if index > self.get_rows_number() or index < 0:
			return None
		# Return copy of the row
		return data[index].copy()

	def get_column(self, index):
		"""Method returns values of required column

		Args:
			index (int): index of column to get values from

		Returns:
			list: values from requested column
			None: if index is incorrect
		"""
		# Check is index valid
		if index > self.get_columns_number() or index < 0:
			return None
		# Return column
		return [row[index] for row in self._data]

	def get_target_column(self):
		"""Method returns list of values from column with target attributes

		Returns:
			list: values from target attributes
		"""
		return self.get_column(self._target_index)

	def get_rows_number(self):
		"""Method returns number of rows in dataset

		Returns:
			int: number of rows
		"""
		return len(self._data)

	def get_columns_number(self):
		"""Method returns number of columns in dataset

		Returns:
			int: number of columns
		"""
		return len(self._data[0])

	def split_by_ratio(self, ratio):
		"""Method split dataset into training and test parts with given ratio

		Args:
			ratio (float): percentage of training part in dataset. In range [0, 1]

		Returns:
			tuple: consists of training and test Dataset
		"""
		# Calculate length of first part of dataset
		TRAIN_DATASET_LEN = round(len(self._data) * ratio)
		# Create list of values for first dataset
		first = [
			self._data[i].copy()
			for i in range(TRAIN_DATASET_LEN)]
		# Create list of values for second dataset
		second = [
			self._data[i].copy()
			for i in range(TRAIN_DATASET_LEN, self.get_rows_number())]
		# Create two datasets and return them
		return (Dataset(first, self._target_index, self._columns_names, self._name),
			Dataset(second, self._target_index, self._columns_names, self._name))

	def shuffle(self):
		"""Method shuffle rows in dataset

		Returns:
			Dataset: Dataset with shuffled rows
		"""
		# Get data
		data = self.data
		# Shuffle data
		random.shuffle(data)
		# Create new dataset on shuffled data
		return Dataset(data, self._target_index, self._columns_names, self._name)

	# TODO different variants of normalization
	# for date type and etc.
	def normalize(self, column_index):
		"""Method normalize the specific column

		Args:
			column_index (int): index of column to normalize
		"""
		# Find minimum value in column
		minimum = min(self.get_column(column_index))
		# Find maximum value in column
		maximum = max(self.get_column(column_index))
		# For each value on column
		for row in self._data:
			# Calculate new value
			row[column_index] = (row[column_index] - minimum) / (maximum - minimum)

	@staticmethod
	def entropy(column):
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

	@staticmethod
	def gain(column, target_column, predicate=None):
		"""Method calculates gain of specific column

		Args:
			index (int): index of column
			predicate (function(x, y)): function that takes two arguments,
				used to group column

		Return:
			float: gain of column
			None: if index is wrong
		"""
		# Concat column and target column
		pairs = zip(column, target_column)
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
		output = Dataset.entropy(target_column)
		# For each group
		for _, value in grouped:
			# Get list of values
			v = list(value)
			# Calculate
			output -= len(v) / rows_number * Dataset.entropy(v)
		# Return entropy
		return output
