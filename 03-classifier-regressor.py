#!/usr/bin/env python3

"""
Machine Learning - Mean squared error, R-squared, Residual Plot

Video:
https://www.youtube.com/watch?v=CbESY3v80zg

Regressors:
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning


How to chose the right algorithm?
By measure mean squared error of regressor!
"""

import numpy as np

from sklearn.metrics import mean_squared_error

# We have right and predicted targets
y_true = np.array([  # Expected outputs
    2,
    1,
    0,
    3,
    2,
    -1,
    5
])

y_pred = np.array([  # Model results
    1,
    1,
    -1,
    3,
    0,
    0,
    4
])


# We need to know the model accuracy
mse = mean_squared_error(y_true, y_pred)
print('MSE', mse)
