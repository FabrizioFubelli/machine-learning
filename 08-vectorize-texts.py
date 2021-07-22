#!/usr/bin/env python3

"""
Machine Learning - Transform texts in numbers

Video:
https://youtu.be/SwcLOvLKAwU

CountVectorizer:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
"""

from sklearn.feature_extraction.text import CountVectorizer

X = [
    'ciao ciao miao',
    'maio',
    'miao bau',
]

vectorizer = CountVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)

print(vectorizer.get_feature_names())
print(X.todense())
print(X)
