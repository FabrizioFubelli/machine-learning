#!/usr/bin/env python3

"""
Machine Learning - Manage null values

Video:
https://youtu.be/QKHYWOu-_lg

SimpleImputer:
https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
"""

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

X = [
    [20, np.nan],
    [np.nan, 'm'],
    [30, 'f'],
    [35, 'f'],
    [np.nan, np.nan],
]

transformers = [
    ['age_imputer', SimpleImputer(), [0]],
    ['sex_imputer', SimpleImputer(strategy='constant', fill_value='n.d.'), [1]],
]

ct = ColumnTransformer(transformers)

X = ct.fit_transform(X)

print(X)
