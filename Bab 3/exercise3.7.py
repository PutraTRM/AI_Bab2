# PCA Breast Cancer

import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets

# Load Breast Cancer data
cancer = datasets.load_breast_cancer()

X = cancer.data
y = cancer.target

# Plot original data
f = plt.figure(1)

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Breast Cancer Data')

plt.show()

# Perform PCA
pca = decomposition.PCA(n_components=3)

pca.fit(X)

X1 = pca.transform(X)

# Plot PCA data
g = plt.figure(2)

plt.scatter(X1[:, 0], X1[:, 1], c=y)

plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Breast Cancer Data')

plt.show()