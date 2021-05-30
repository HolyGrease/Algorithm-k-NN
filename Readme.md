# Example of creating Dataset object
Firstly, import Dataset:

	from dataset import Dataset

Secondly, get some data:

	data = [
		[5.1, 3.5, 1.4, 0.2, "Iris-setosa"],
		[5.0, 3.2, 1.2, 0.2, "Iris-setosa"],
		[6.4, 3.2, 4.5, 1.5, "Iris-versicolor"],
		[6.7, 3.1, 4.4, 1.4, "Iris-versicolor"],
		[6.7, 3.0, 5.2, 2.3, "Iris-virginica"]]

Thirdly, set columns (attributes) names:

	column_names = [
		"Sepal length", "Sepal width",
		"Petal length", "Petal width",
		"Class"]

Now we can create Dataset object. Arguments:
- data - just list of list
- target index - index of target attribute, attribute that contains classes values
- column or attributes names - list of attributes names
- name - Dataset name


	iris = Dataset(data, 4, column_names, "Iris")

Also you can just get iris dataset by calling method

	get_iris().

You can specify path to dataset file by passing this path as argument, for example:

	get_iris("data\\iris.data")

Default value of path 
> resources\\data\\iris\\iris.data.

	iris = get_iris()

# Preprocessing dataset
Shuffle dataset:

	iris = iris.shuffle()

Split dataset on "train" and "test", as argument passing ratio. Train dataset gets 80% of original dataset, test - other:

	train, test = iris.split_by_ration(0.8)

# Classification
Don't forget to import, instead of euclidean you can import any other implemented metric:

	from k_NN import k_NN
	from k_NN import euclidean

### Basic classification
For basic classification use k_NN() function with only 4 arguments:
- dataset - train dataset which used for classification
- k - integer, number of neighbours that will be used to predict class
- row - instance to classify, must contains all attributes except target
- metric - metric used to calculate distance. k_NN.py file contains some metric, use one of them or implement yours.
k_NN method return class according to train dataset

Remember you need to delete target attribute from instance that you classify

	assert_class = test.data[0].pop(test.target)
	instance_to_classify = [4.8, 3.1, 1.6, 0.2]
	predicted_class = k_NN(train, instance_to_classify, 3, euclidean)
	print(f"{assert_class} ?= {predicted_class}")

In terminal you can see something like this
> Iris-setosa ?= Iris-setosa
### Attribute weight classification
Attribute weights are used with calculating distances. This weights determine how specific attribute influece on general distance.
In this case you need to define weights. For example like this:

	attributes_weights = [0.1, 0.5, 0.6, 0.3]

Number of weights must be equal number of attributes - 1 (except target attribute)
Also you can use build in Dataset method called [inforamtion gain](https://machinelearningmastery.com/information-gain-and-mutual-information/#:~:text=Information%20gain%20is%20the%20reduction,before%20and%20after%20a%20transformation.) to get data based weights. Example:

	attributes_weights = [
		Dataset.gain(
			train.get_column(i),
			train.get_target_column())
		for i in range(4)]

To use this weights in classification you need pass one more argument:

	predicted_class = k_NN(train, row, 3, euclidean, attributes_weights=attributes_weights)

### Distance weight classification
In this case you need to define weights. For example like this:

	distances_weights = [0.1, 0.5, 0.6, 0.3]

To use this weights in classification you need pass one more argument:

	predicted_class = k_NN(train, row, 3, euclidean, distances_weights)
### Combine weight classification
Also you can combine both, attribute and distance weights, in this algorithm.

	predicted_class = k_NN(train, row, 3, euclidean, distances_weights, attributes_weights)