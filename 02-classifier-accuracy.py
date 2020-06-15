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

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

# We have right and predicted targets
y_true = [  # Expected outputs
    'rugby',
    'rugby',
    'calcio',
    'basket',
    'basket',
    'calcio',
    'basket'
]

y_pred = [  # Model results
    'rugby',
    'rugby',
    'basket',
    'basket',
    'rugby',
    'calcio',
    'basket'
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
c_matrix_display = ConfusionMatrixDisplay(
    c_matrix,
    display_labels=[    # Alphabetically sort
        'basket',
        'calcio',
        'rugby'
    ]
)
c_matrix_display.plot(include_values=True)
plt.show()
