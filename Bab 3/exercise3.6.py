from sklearn.datasets import make_classification
from sklearn.decomposition import PCA

X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False
)

print(X)

clf = PCA()

clf.fit(X, y)

print(clf.explained_variance_ratio_)

print(clf.singular_values_)