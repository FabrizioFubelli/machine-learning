#!/usr/bin/env python3

"""
Machine Learning - Mean squared error, R-squared, Residual Plot

Video:
https://www.youtube.com/watch?v=CbESY3v80zg

Regressors:
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning


How to chose the right algorithm?
By measuring mean squared error of regressor!
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

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


# We need to know the model mean squared error
mse = mean_squared_error(y_true, y_pred)
print('MSE', mse)

# R² = ratio between the "mean squared error" of an imaginary model
#      and our mse

# The imaginary model is a model which answer with a constant value foreach
# dataset record. The constant is equals to the mean of desired answers.
# The mean squared error of desired answers is called "baseline".

# R² = 1 - (MSE/MSE_base) = [-∞, +1]
r2 = r2_score(y_true, y_pred)
print('r2', r2)

residuals = y_pred - y_true

plt.scatter(y_true, residuals)
plt.xlabel('True')
plt.ylabel('Error')

plt.show()

# By finding "patterns" inside the graph,
# it's possible to find improvable data in features
