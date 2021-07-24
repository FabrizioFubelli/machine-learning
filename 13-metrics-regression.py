#!/usr/bin/env python3

"""
Machine Learning - Measure Regression Accuracy

Video:
https://youtu.be/r9bLO6b7bZg
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y = np.random.random(size=100) * 10
# errors = + (2 * (np.random.random(size=100)) - 1)           # random
# errors = + y/2 + (2 * (np.random.random(size=100)) - 1)     # link errors to target
errors = + y**2 * (2 * (np.random.random(size=100)) - 1)      # link errors to target
p = y + errors

mse = mean_squared_error(y, p)
mae = mean_absolute_error(y, p)

res = y - p
sns.scatterplot(x=y, y=res)
plt.show()
