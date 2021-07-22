#!/usr/bin/env python3

"""
Machine Learning - Confusion Matrix

Video:
https://youtu.be/9G3l6c2EIfM


How to measure the performance of a classifier?
Thanks to Accuracy and Confusion Matrix!
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_confusion_matrix

y_test = np.array([  # Expected outputs
    'basket',
    'rugby',
    'rugby',
    'calcio',
    'basket',
    'basket',
    'calcio',
])

y_pred = np.array([  # Model results
    'basket',
    'rugby',
    'rugby',    # calcio
    'basket',
    'basket',
    'rugby',
    'calcio',
])

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy', accuracy)

plot_confusion_matrix(y_test, y_pred)
plt.show()
