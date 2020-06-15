#!/usr/bin/env python3

"""
Machine Learning - Classifier Accuracy and Confusion Matrix

Video:
https://www.youtube.com/watch?v=9G3l6c2EIfM

Classifiers:
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning


How to chose the right algorithm?
By measure accuracy of classifier!
"""

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

# We have right and predicted targets
y_true = [  # Expected outputs
    'basket',
    'rugby',
    'rugby',
    'calcio',
    'basket',
    'basket',
    'calcio'
]

y_pred = [  # Model results
    'basket',
    'rugby',
    'rugby',
    'basket',
    'basket',
    'rugby',
    'calcio'
]


# We need to know the model accuracy
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy', str(accuracy))

# Now we have certain accuracy.

# How to increase this accuracy?
# By analyzing the wrong results!

# We could improve the input dataset or elaborate the features.

# A good tool to check the results, is the "Confusion Matrix"

c_matrix = confusion_matrix(y_true, y_pred)
plt.show()
