#!/usr/bin/env python3

"""
Machine Learning - Measure Classifier Accuracy

Video:
https://youtu.be/DJeh_CJpvAw

KNeighborsClassifier:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""

import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def randomize(v, lab, prob=0.2):
    v2 = []
    for el in v:
        if np.random.random() > prob:
            v2.append(el)
        else:
            v2.append(np.random.choice(lab))
    return v2


labels = ['cronaca', 'politica', 'sport']
y = np.random.choice(labels, 1000)
p = randomize(y, labels)

acc = accuracy_score(y, p)
f1_sc = f1_score(y, p, average='weighted')
print(f'Accuracy {acc}')
print(f'Misclassification {1 - acc}')
print(f'F1 Score {f1_sc}')

report = classification_report(y, p)
print(report)

skplt.metrics.plot_confusion_matrix(y, p)
plt.show()
