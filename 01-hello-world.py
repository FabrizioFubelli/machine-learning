#!/usr/bin/env python3

"""
Machine Learning - Hello World

Video:
https://www.youtube.com/watch?v=hSZH6saoLBY

1) Analyze input data
2) Split features and target
3) Split learning data and test data
4) Execute learning with learning data
5) Predict result of learning data and test data
6) Compare the accuracy scores between learning and test data
"""

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# The input data is an iris flower dataset.
# The desired output is the class of flower, by analyzing
# the following parameters:
#  - Sepal length
#  - Sepal width
#  - Petal length
#  - Petal width

iris_dataset = datasets.load_iris()

X = iris_dataset.data       # Features
y = iris_dataset.target     # Target

print(iris_dataset['DESCR'])
print()

# Split data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# The test data is not used during learning, but is needed to measure
# the final model learning quality

# Execute learning
model = DecisionTreeClassifier()
model.fit(X_train, y_train)


predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test)

print('Train accuracy')
print(accuracy_score(y_train, predicted_train))

print('Test score')
print(accuracy_score(y_test, predicted_test))
