#!/usr/bin/env python3

"""
Machine Learning - Train a predictive model with classification (category target)

Video:
https://youtu.be/lMJqXncEn78
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dataset = load_iris()

# X contains the features
X = dataset['data']
# y contains the target we want
y = dataset['target']

print(X[0], y[0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)         # Train model from data

p_train = model.predict(X_train)    # Predict X_train after training
p_test = model.predict(X_test)      # Predict X_test after training

acc_train = accuracy_score(y_train, p_train)
acc_test = accuracy_score(y_test, p_test)
print(f'Accuracy train: {acc_train}')
print(f'Accuracy test: {acc_test}')
