from dataset import Dataset
from k_NN import k_NN
from k_NN import euclidean

def main():
	# Load dataset
	iris = Dataset.get_iris()

	iris.print(10)
	# Shuffle dataset
	iris = iris.shuffle()
	# Split dataset on train and test
	# train dataset get 0.8 ratio of original dataset
	train, test = iris.split_by_ratio(0.8)
	# Variable to count correct predictions

	assert_class = test.data[0].pop(test.target)
	instance_to_classify = [4.8, 3.1, 1.6, 0.2]
	predicted_class = k_NN(train, 3, instance_to_classify, euclidean)
	print(f"{assert_class} ?= {predicted_class}")


	correct = 0
	for row in test.data:
		# Get correct value
		assert_value = row.pop(test.target)
		# Make prediction
		predicted_class = k_NN(train, 3, row, euclidean)
		# If prediction is correct
		if predicted_class == assert_value:
			correct += 1
		# Print compare log to termainal
		print("{:<15} ?= {:<15}".format(assert_value, predicted_class))
	# Count and print accuracy
	print("Acurracy: {:1.2}".format(correct / test.get_rows_number()))

if __name__ == '__main__':
	main()