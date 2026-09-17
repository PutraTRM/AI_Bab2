# EXAMPLE 3.28D THE LAZYPREDICT.IPYNB PROGRAM (PART 4)

import lazypredict

from lazypredict.Supervised import LazyRegressor

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import numpy as np

from sklearn.datasets import fetch_california_housing

# Dataset
X, y = fetch_california_housing(
    return_X_y=True,
    as_frame=True
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=.1,
    random_state=1
)

# LazyPredict
reg = LazyRegressor()

models, predictions = reg.fit(
    X_train,
    X_test,
    y_train,
    y_test
)

# Tampilkan hasil
print(models)