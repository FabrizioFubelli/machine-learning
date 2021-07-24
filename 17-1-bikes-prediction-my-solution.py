#!/usr/bin/env python3

"""
Machine Learning - Predict bike rents (Fabrizio Fubelli approach)

Video:
https://youtu.be/pj3ZX7-1v1c
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

np.random.seed(0)

df = pd.read_csv('files/bike-sharing/hour.csv')
X = df.drop(['cnt', 'dteday', 'instant'], axis=1)
y = df['cnt'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = MLPRegressor(hidden_layer_sizes=[2], max_iter=25, tol=-1, verbose=2)

model.fit(X_train, y_train)

p_train = model.predict(X_train)
p_test = model.predict(X_test)
p = model.predict(X)

mae_train = mean_absolute_error(y_train, p_train)
mae_test = mean_absolute_error(y_test, p_test)

print(f'Train {mae_train}, test {mae_test}')

sns.scatterplot(x=X_train['hr'].values, y=y_train)
sns.scatterplot(x=X_test['hr'].values, y=y_test)
sns.lineplot(x=X['hr'].values, y=p)
plt.show()
