#!/usr/bin/env python3

"""
Machine Learning - Scale the features

Video:
https://youtu.be/j3Csuz26heo

KNeighborsClassifier:
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataset = load_wine()

print(dataset['DESCR'])

X = dataset['data'][:, [4, 7]]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

df = pd.DataFrame(X, columns=['magnesium', 'phenols'])
g = sns.scatterplot(data=df, x='magnesium', y='phenols')
# g.set(xlim=(-10, 200), ylim=(-10, 200))
plt.show()

# Non-Scaled Training

X = dataset['data']
y = dataset['target']

model = KNeighborsClassifier()
model.fit(X, y)

p = model.predict(X)

acc_not_scaled = accuracy_score(y, p)
print(f'Accuracy not scaled {acc_not_scaled}')

# Scaled Training

X = dataset['data']
y = dataset['target']

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

model2 = KNeighborsClassifier()
model2.fit(X, y)

p = model2.predict(X)

acc_scaled = accuracy_score(y, p)
print(f'Accuracy scaled {acc_scaled}')
