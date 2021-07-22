#!/usr/bin/env python3

"""
Machine Learning - Train a predictive model with regression

Video:
https://youtu.be/7YDWaTKtCdI

LinearRegression:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
"""

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import numpy as np

np.random.seed(2)

dataset = load_boston()

# X contains the features
X = dataset['data']
# y contains the target we want to find
y = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)         # Train model from data

p_train = model.predict(X_train)    # Predict X_train after training
p_test = model.predict(X_test)      # Predict X_test after training

mae_train = mean_absolute_error(y_train, p_train)
mae_test = mean_absolute_error(y_test, p_test)
print('MAE train', mae_train)
print('MAE test', mae_test)

# We need to know the model mean squared error
mse_train = mean_squared_error(y_train, p_train)
mse_test = mean_squared_error(y_test, p_test)
print('MSE train', mse_train)
print('MSE test', mse_test)
